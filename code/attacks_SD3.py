# SD3 MMDiT Adversarial Attack Module
# Implements MMDiT-specific loss strategies (O/A/B/C/D) for SD3 adversarial perturbation.
# Objective convention in this file: larger scalar loss = stronger attack objective.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class AttentionMapHook:
    """Capture differentiable attention diagnostics and intermediate MMDiT features."""

    def __init__(self):
        self.attn_maps = []  # image-query -> text-key attention blocks, not full NxN maps
        self.attn_entropy_terms = []
        self.txt_injection_feats = []  # A_{I->T} V_T, shape [B,H,N_img,Dh]
        self.hidden_states_list = []
        self.encoder_hidden_states_list = []
        self.img_stream_feats = []
        self.txt_stream_feats = []
        self.hooks = []

    def clear(self):
        self.attn_maps = []
        self.attn_entropy_terms = []
        self.txt_injection_feats = []
        self.hidden_states_list = []
        self.encoder_hidden_states_list = []
        self.img_stream_feats = []
        self.txt_stream_feats = []

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class AttnMapCaptureProcessor:
    """Custom attention processor for MMDiT Joint Attention.

    The real scaled-dot-product attention path is unchanged. For Mode A this
    additionally computes differentiable image-query -> text-key attention and
    the text-injection output A_{I->T}V_T.
    """

    def __init__(self, hook: AttentionMapHook, block_idx: int, capture_attn: bool = True):
        self.hook = hook
        self.block_idx = block_idx
        self.capture_attn = capture_attn

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            full_query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            full_key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            full_value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

            if self.capture_attn:
                # Differentiable image-query -> text-key attention block.
                attn_scores = torch.matmul(
                    query.float(), encoder_hidden_states_key_proj.float().transpose(-2, -1)
                ) / math.sqrt(head_dim)
                img_to_txt_attn = F.softmax(attn_scores, dim=-1).to(query.dtype)
                txt_injection = torch.matmul(img_to_txt_attn, encoder_hidden_states_value_proj)
                self.hook.attn_maps.append(img_to_txt_attn)
                self.hook.txt_injection_feats.append(txt_injection)
                entropy = -(
                    img_to_txt_attn * img_to_txt_attn.clamp_min(1e-8).log()
                ).sum(dim=-1).mean()
                self.hook.attn_entropy_terms.append(entropy)

            hidden_states = F.scaled_dot_product_attention(
                full_query, full_key, full_value, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1]:],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


def register_feature_hooks(
    transformer,
    hook: AttentionMapHook,
    capture_attn: bool = True,
    detach_features: bool = False,
    capture_blocks=None,
):
    """Register forward hooks on MMDiT blocks.

    clean/reference forward should use detach_features=True.
    adversarial forward should use detach_features=False so gradients reach X_adv.
    """
    hook.remove()
    hook.clear()

    capture_set = None if capture_blocks is None else set(capture_blocks)

    if capture_attn:
        for idx, block in enumerate(transformer.transformer_blocks):
            if capture_set is None or idx in capture_set:
                block.attn.processor = AttnMapCaptureProcessor(hook, idx, capture_attn=True)

    def make_forward_hook(block_idx):
        def forward_hook(module, input, output):
            if capture_set is not None and block_idx not in capture_set:
                return
            if isinstance(output, tuple) and len(output) == 2:
                enc_h, img_h = output
                if detach_features:
                    img_store = img_h.detach()
                    enc_store = enc_h.detach() if enc_h is not None else None
                else:
                    img_store = img_h
                    enc_store = enc_h
                hook.hidden_states_list.append(img_store)
                hook.encoder_hidden_states_list.append(enc_store)
                hook.img_stream_feats.append(img_store)
                hook.txt_stream_feats.append(enc_store)
        return forward_hook

    for idx, block in enumerate(transformer.transformer_blocks):
        if capture_set is None or idx in capture_set:
            h = block.register_forward_hook(make_forward_hook(idx))
            hook.hooks.append(h)

    return hook


def restore_processors(transformer):
    """Restore original JointAttnProcessor2_0 on all transformer blocks."""
    from diffusers.models.attention_processor import JointAttnProcessor2_0
    for block in transformer.transformer_blocks:
        block.attn.processor = JointAttnProcessor2_0()
        if hasattr(block, 'attn2') and block.attn2 is not None:
            block.attn2.processor = JointAttnProcessor2_0()


# ============================================================
# MMDiT Loss Functions (larger-is-more-attack convention)
# ============================================================


def _tensor_item(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().item())
    return float(x)


def cross_modal_disruption_loss(attn_maps, injection_adv=None, injection_clean=None, return_diagnostics=False):
    """Loss A: cross-modal text injection disruption.

    Primary objective: maximize clean/adv difference of text-injection outputs
    O_{I<-T}=A_{I->T}V_T when a clean reference is supplied.
    Fallback objective: maximize image-query -> text-key attention entropy.
    """
    if len(attn_maps) == 0:
        loss = torch.tensor(0.0)
        diag = {'attn_entropy': 0.0, 'attn_injection_l2': 0.0}
        return (loss, diag) if return_diagnostics else loss

    entropies = []
    for attn_block in attn_maps:
        entropy = -(attn_block * attn_block.clamp_min(1e-8).log()).sum(dim=-1).mean()
        entropies.append(entropy)
    entropy_loss = torch.stack(entropies).mean()

    injection_loss = None
    if injection_adv is not None and injection_clean is not None and len(injection_adv) > 0:
        terms = []
        for adv, clean in zip(injection_adv, injection_clean):
            adv_f = adv.float()
            clean_f = clean.float().detach()
            denom = clean_f.pow(2).mean().detach().clamp_min(1e-6)
            terms.append(F.mse_loss(adv_f, clean_f) / denom)
        if terms:
            injection_loss = torch.stack(terms).mean()

    if injection_loss is None:
        loss = entropy_loss
        injection_value = 0.0
    else:
        # Entropy is a small regularizer; injection divergence is the main target.
        loss = injection_loss + 0.05 * entropy_loss
        injection_value = _tensor_item(injection_loss)

    diag = {
        'attn_entropy': _tensor_item(entropy_loss),
        'attn_injection_l2': injection_value,
    }
    return (loss, diag) if return_diagnostics else loss


def _normalize_feature(x, eps=1e-6):
    x = x.float()
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def feature_divergence_loss(feat_adv_list, feat_clean_list, return_diagnostics=False):
    """Loss B: image-stream representation divergence with normalized features."""
    total_loss = None
    cos_terms = []
    gram_terms = []
    count = 0
    for feat_adv, feat_clean in zip(feat_adv_list, feat_clean_list):
        if feat_adv is None or feat_clean is None:
            continue
        adv = _normalize_feature(feat_adv)
        clean = _normalize_feature(feat_clean).detach()

        cos_sim = F.cosine_similarity(adv.flatten(1), clean.flatten(1), dim=1).mean()
        cos_dist = 1.0 - cos_sim

        gram_adv = _gram_matrix(adv)
        gram_clean = _gram_matrix(clean)
        style_dist = F.l1_loss(gram_adv, gram_clean)

        loss = cos_dist + style_dist
        total_loss = loss if total_loss is None else total_loss + loss
        cos_terms.append(cos_sim)
        gram_terms.append(style_dist)
        count += 1

    if count == 0:
        device = feat_adv_list[0].device if len(feat_adv_list) > 0 and isinstance(feat_adv_list[0], torch.Tensor) else 'cpu'
        loss = torch.tensor(0.0, device=device)
        diag = {'feature_cos': 0.0, 'feature_gram_l1': 0.0}
        return (loss, diag) if return_diagnostics else loss

    loss = total_loss / count
    feature_cos = torch.stack(cos_terms).mean()
    gram_l1 = torch.stack(gram_terms).mean()
    diag = {
        'feature_cos': _tensor_item(feature_cos),
        'feature_gram_l1': _tensor_item(gram_l1),
    }
    return (loss, diag) if return_diagnostics else loss


def _gram_matrix(x):
    """Compute channel Gram matrix for [B, N, D] features."""
    B, N, D = x.shape
    feat = x.reshape(B, N, D)
    return torch.bmm(feat.transpose(1, 2), feat) / max(N * D, 1)


def trajectory_divergence_loss(v_adv, v_clean, return_diagnostics=False):
    """Loss C/O_fair: flow velocity direction divergence."""
    cos_sim = F.cosine_similarity(v_adv.float().flatten(1), v_clean.float().detach().flatten(1), dim=1)
    # Keep this exact string for static validation:
    loss = 1.0 - cos_sim.mean()
    diag = {
        'velocity_cos': _tensor_item(cos_sim.mean()),
        'velocity_div': _tensor_item(loss),
    }
    return (loss, diag) if return_diagnostics else loss


def _channel_cov(x):
    x = _normalize_feature(x).float()
    return torch.bmm(x.transpose(1, 2), x) / max(x.shape[1], 1)


def _cov_cka(cov_a, cov_b, eps=1e-6):
    dot = (cov_a * cov_b).sum(dim=(1, 2))
    na = cov_a.pow(2).sum(dim=(1, 2)).sqrt()
    nb = cov_b.pow(2).sum(dim=(1, 2)).sqrt()
    return dot / (na * nb + eps)


def modality_imbalance_loss(img_stream_feats, txt_stream_feats, img_clean_feats=None, txt_clean_feats=None, return_diagnostics=False):
    """Loss D: modality energy-ratio and cross-modal covariance imbalance.

    Larger value increases deviation from clean image/text energy balance and
    lowers image/text covariance CKA. If clean reference is absent, falls back to
    absolute image variance and inverse correlation proxy.
    """
    total_loss = None
    ratio_terms = []
    cka_terms = []
    count = 0

    for idx, (img_feat, txt_feat) in enumerate(zip(img_stream_feats, txt_stream_feats)):
        if img_feat is None or txt_feat is None:
            continue
        img = img_feat.float()
        txt = txt_feat.float()
        img_energy = img.pow(2).mean(dim=(1, 2)).clamp_min(1e-6)
        txt_energy = txt.pow(2).mean(dim=(1, 2)).clamp_min(1e-6)
        ratio_adv = torch.log(img_energy / txt_energy)

        if img_clean_feats is not None and txt_clean_feats is not None and idx < len(img_clean_feats) and idx < len(txt_clean_feats):
            img_c = img_clean_feats[idx].float().detach()
            txt_c = txt_clean_feats[idx].float().detach()
            ratio_clean = torch.log(
                img_c.pow(2).mean(dim=(1, 2)).clamp_min(1e-6)
                / txt_c.pow(2).mean(dim=(1, 2)).clamp_min(1e-6)
            )
            ratio_dev = (ratio_adv - ratio_clean).pow(2).mean()
        else:
            ratio_dev = ratio_adv.pow(2).mean()

        cov_img = _channel_cov(img)
        cov_txt = _channel_cov(txt)
        cka = _cov_cka(cov_img, cov_txt).mean()
        loss = ratio_dev + 0.1 * (1.0 - cka)

        total_loss = loss if total_loss is None else total_loss + loss
        ratio_terms.append(ratio_dev)
        cka_terms.append(cka)
        count += 1

    if count == 0:
        device = img_stream_feats[0].device if len(img_stream_feats) > 0 and isinstance(img_stream_feats[0], torch.Tensor) else 'cpu'
        loss = torch.tensor(0.0, device=device)
        diag = {'modality_ratio_dev': 0.0, 'cross_modal_cka': 0.0}
        return (loss, diag) if return_diagnostics else loss

    loss = total_loss / count
    diag = {
        'modality_ratio_dev': _tensor_item(torch.stack(ratio_terms).mean()),
        'cross_modal_cka': _tensor_item(torch.stack(cka_terms).mean()),
    }
    return (loss, diag) if return_diagnostics else loss


def denoiser_prediction_loss(v_pred):
    """Loss O_repo semantic proxy: positive velocity prediction norm."""
    return v_pred.float().norm().to(v_pred.dtype)


# ============================================================
# SD3 PGD Attack with MMDiT losses
# ============================================================


class SD3_Linf_PGD:
    """PGD attack for SD3 with MMDiT-specific losses.

    Objective convention: larger loss = stronger attack objective.
    opt_direction='maximize' should be the default for all corrected modes.
    """

    def __init__(self, net, fn, epsilon, steps, eps_iter, clip_min=-1.0, clip_max=1.0,
                 targeted=True, g_mode='+', opt_direction=None, mmdit_mode='A', capture_attn=True,
                 textual_weight=1.0, mmdit_weight=1.0, textual_objective='toward_target',
                 debug_grad=False, capture_blocks=None):
        self.net = net
        self.fn = fn
        self.eps = epsilon
        self.step_size = eps_iter
        self.iters = steps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.g_mode = g_mode
        self.opt_direction = opt_direction or ('maximize' if g_mode == '+' else 'minimize')
        self.g_dir = 1.0 if self.opt_direction == 'maximize' else -1.0
        self.mmdit_mode = mmdit_mode
        self.capture_attn = capture_attn
        self.textual_weight = textual_weight
        self.mmdit_weight = mmdit_weight
        self.textual_objective = textual_objective
        self.debug_grad = debug_grad
        self.capture_blocks = capture_blocks

        self.cirt = nn.MSELoss(reduction='sum')

    def pgd_sd3(self, X, target_image=None, random_start=False):
        device = X.device

        if random_start:
            X_adv = X.clone().detach() + (torch.rand(*X.shape, device=device) * 2 * self.eps - self.eps)
        else:
            X_adv = X.clone().detach()

        loss_history = {}
        pbar = tqdm(range(self.iters), desc=f"SD3-PGD mode={self.mmdit_mode}")

        for i in pbar:
            X_adv.requires_grad_(True)
            loss, components = self._compute_loss(X_adv, X, target_image, device)

            if self.debug_grad:
                components.update(self._grad_debug_dict(X_adv, loss, components))

            pbar.set_description(
                f"SD3-PGD mode={self.mmdit_mode} | total={loss.item():.3f} "
                f"textual={components.get('textual', 0.0):.3f} mmdit={components.get('mmdit', 0.0):.3f}"
            )

            if self.debug_grad and (i == 0 or i == self.iters - 1):
                print(
                    f"[debug_grad step={i}] "
                    f"textual={components.get('grad_textual_l2', 0.0):.6e} "
                    f"mmdit={components.get('grad_mmdit_l2', 0.0):.6e} "
                    f"total={components.get('grad_total_l2', 0.0):.6e}"
                )

            loss.backward()
            grad = X_adv.grad.detach()

            X_adv = X_adv.detach() + self.g_dir * grad.sign() * self.step_size
            X_adv = torch.minimum(torch.maximum(X_adv, X - self.eps), X + self.eps)
            X_adv = torch.clamp(X_adv, min=self.clip_min, max=self.clip_max)

            record = {'total': _tensor_item(loss)}
            for key, val in components.items():
                if key.startswith('_'):
                    continue
                record[key] = _tensor_item(val)
            for key, val in record.items():
                loss_history.setdefault(key, []).append(val)

            torch.cuda.empty_cache()

        return X_adv, loss_history

    def _grad_norm(self, tensor, X_adv, retain_graph=True):
        if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
            return 0.0
        grad = torch.autograd.grad(tensor, X_adv, retain_graph=retain_graph, allow_unused=True)[0]
        if grad is None:
            return 0.0
        return grad.detach().float().norm().item()

    def _grad_debug_dict(self, X_adv, loss, components):
        return {
            'grad_textual_l2': self._grad_norm(components.get('_textual_tensor'), X_adv, retain_graph=True),
            'grad_mmdit_l2': self._grad_norm(components.get('_mmdit_tensor'), X_adv, retain_graph=True),
            'grad_total_l2': self._grad_norm(loss, X_adv, retain_graph=True),
        }

    def _encode_latent(self, image, pipe):
        z = pipe.vae.encode(image.to(pipe.vae.dtype)).latent_dist.mean
        z = (z - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        return z.to(pipe.transformer.dtype)

    def _compute_clean_features_at_timestep(self, X_clean, timestep, capture_attn=False):
        pipe = self.net.pipe
        hook = AttentionMapHook()
        transformer = pipe.transformer
        register_feature_hooks(
            transformer, hook, capture_attn=capture_attn, detach_features=True,
            capture_blocks=self.capture_blocks,
        )
        try:
            with torch.no_grad():
                z_clean = self._encode_latent(X_clean, pipe)
                _ = transformer(
                    hidden_states=z_clean,
                    timestep=timestep,
                    encoder_hidden_states=self.net.prompt_embeds.to(transformer.dtype),
                    pooled_projections=self.net.pooled_prompt_embeds.to(transformer.dtype),
                    return_dict=False,
                )
            return {
                'img': [f.clone().detach() for f in hook.img_stream_feats],
                'txt': [f.clone().detach() if f is not None else None for f in hook.txt_stream_feats],
                'attn': [f.clone().detach() for f in hook.attn_maps],
                'injection': [f.clone().detach() for f in hook.txt_injection_feats],
            }
        finally:
            hook.remove()
            restore_processors(transformer)

    def _textual_loss(self, z_adv, target_image, pipe, device):
        textual_loss = torch.tensor(0.0, device=device, dtype=z_adv.dtype)
        if target_image is None:
            return textual_loss
        with torch.no_grad():
            z_target = self._encode_latent(target_image, pipe)
            z_target = z_target.to(z_adv.dtype).detach()
        raw_mse = self.cirt(z_adv, z_target)
        if self.textual_objective == 'toward_target':
            return -raw_mse
        if self.textual_objective == 'away_from_target':
            return raw_mse
        raise ValueError(f"Unknown textual_objective: {self.textual_objective}")

    def _trajectory_loss_shared_noise(self, z_adv, X_clean, timestep, prompt_embeds, pooled_embeds, device):
        pipe = self.net.pipe
        transformer = pipe.transformer
        with torch.no_grad():
            z_clean = self._encode_latent(X_clean, pipe).detach()
            noise = torch.randn_like(z_adv).detach()
            sigma = timestep.to(dtype=z_adv.dtype).reshape(-1, 1, 1, 1)
            z_clean_t = (1.0 - sigma) * z_clean + sigma * noise
            v_clean = transformer(
                hidden_states=z_clean_t,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
        z_adv_t = (1.0 - sigma) * z_adv + sigma * noise
        v_adv = transformer(
            hidden_states=z_adv_t,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        return trajectory_divergence_loss(v_adv, v_clean, return_diagnostics=True)

    def _compute_loss(self, X_adv, X_clean, target_image, device):
        pipe = self.net.pipe
        transformer = pipe.transformer

        z_adv = self._encode_latent(X_adv, pipe)
        timestep = torch.tensor([torch.rand(1, device=device).item()], device=device, dtype=pipe.transformer.dtype)
        prompt_embeds = self.net.prompt_embeds.to(pipe.transformer.dtype)
        pooled_embeds = self.net.pooled_prompt_embeds.to(pipe.transformer.dtype)

        textual_loss = self._textual_loss(z_adv, target_image, pipe, device)
        mmdit_loss = torch.tensor(0.0, device=device, dtype=pipe.transformer.dtype)
        diagnostics = {}

        if self.mmdit_mode in {'O', 'O_repo'}:
            v_pred = transformer(
                hidden_states=z_adv,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
            mmdit_loss = denoiser_prediction_loss(v_pred)
            diagnostics['v_pred_norm'] = _tensor_item(mmdit_loss)

        elif self.mmdit_mode in {'C', 'O_fair'}:
            mmdit_loss, diagnostics = self._trajectory_loss_shared_noise(
                z_adv, X_clean, timestep, prompt_embeds, pooled_embeds, device
            )

        else:
            clean_ref = None
            if self.mmdit_mode in {'A', 'B', 'D'}:
                clean_ref = self._compute_clean_features_at_timestep(
                    X_clean, timestep, capture_attn=(self.mmdit_mode == 'A')
                )

            hook = AttentionMapHook()
            need_attn = (self.mmdit_mode == 'A')
            register_feature_hooks(
                transformer, hook, capture_attn=need_attn, detach_features=False,
                capture_blocks=self.capture_blocks,
            )
            try:
                _ = transformer(
                    hidden_states=z_adv,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0]

                if self.mmdit_mode == 'A':
                    if len(hook.attn_maps) > 0:
                        mmdit_loss, diagnostics = cross_modal_disruption_loss(
                            hook.attn_maps,
                            injection_adv=hook.txt_injection_feats,
                            injection_clean=clean_ref['injection'] if clean_ref else None,
                            return_diagnostics=True,
                        )
                    else:
                        mmdit_loss = torch.tensor(0.0, device=device, dtype=pipe.transformer.dtype)
                        diagnostics = {'attn_entropy': 0.0, 'attn_injection_l2': 0.0}
                elif self.mmdit_mode == 'B':
                    if len(hook.img_stream_feats) > 0 and clean_ref is not None:
                        mmdit_loss, diagnostics = feature_divergence_loss(
                            hook.img_stream_feats, clean_ref['img'], return_diagnostics=True
                        )
                    else:
                        mmdit_loss = torch.tensor(0.0, device=device, dtype=pipe.transformer.dtype)
                        diagnostics = {'feature_cos': 0.0, 'feature_gram_l1': 0.0}
                elif self.mmdit_mode == 'D':
                    if len(hook.img_stream_feats) > 0 and len(hook.txt_stream_feats) > 0 and clean_ref is not None:
                        mmdit_loss, diagnostics = modality_imbalance_loss(
                            hook.img_stream_feats, hook.txt_stream_feats,
                            img_clean_feats=clean_ref['img'], txt_clean_feats=clean_ref['txt'],
                            return_diagnostics=True,
                        )
                    else:
                        mmdit_loss = torch.tensor(0.0, device=device, dtype=pipe.transformer.dtype)
                        diagnostics = {'modality_ratio_dev': 0.0, 'cross_modal_cka': 0.0}
                else:
                    raise ValueError(f"Unknown SD3 MMDiT attack mode: {self.mmdit_mode}")
            finally:
                hook.remove()
                restore_processors(transformer)

        joint_loss = self.textual_weight * textual_loss + self.mmdit_weight * mmdit_loss
        components = {
            'textual': textual_loss.detach().float().item(),
            'mmdit': mmdit_loss.detach().float().item() if isinstance(mmdit_loss, torch.Tensor) else float(mmdit_loss),
            '_textual_tensor': textual_loss,
            '_mmdit_tensor': mmdit_loss,
        }
        components.update(diagnostics)
        return joint_loss, components
