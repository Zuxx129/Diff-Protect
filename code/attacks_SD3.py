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
        self.hidden_states_list = []
        self.encoder_hidden_states_list = []
        self.img_stream_feats = []
        self.txt_stream_feats = []
        self.hooks = []

    def clear(self):
        self.attn_maps = []
        self.attn_entropy_terms = []
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

    It keeps the real scaled-dot-product attention path unchanged, but additionally
    computes a differentiable image-query -> text-key attention diagnostic for
    Mode A. We intentionally avoid storing full [B,H,N,N] attention maps.
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
                # This is a targeted cross-modal diagnostic, not the full joint NxN attention map.
                attn_scores = torch.matmul(
                    query.float(), encoder_hidden_states_key_proj.float().transpose(-2, -1)
                ) / math.sqrt(head_dim)
                img_to_txt_attn = F.softmax(attn_scores, dim=-1).to(query.dtype)
                self.hook.attn_maps.append(img_to_txt_attn)
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


def _zero_like_device(items, fallback='cpu'):
    for item in items:
        if isinstance(item, torch.Tensor):
            return torch.tensor(0.0, device=item.device, dtype=item.dtype)
    return torch.tensor(0.0, device=fallback)


def cross_modal_disruption_loss(attn_maps, N_img_tokens=None):
    """Loss A: Cross-modal injection disruption.

    Larger value means more diffuse / less selective image-query -> text-key attention.
    The current implementation optimizes entropy as a first-order proxy and records
    differentiable attention blocks produced by AttnMapCaptureProcessor.
    """
    if len(attn_maps) == 0:
        return torch.tensor(0.0)

    total_loss = None
    count = 0
    for attn_block in attn_maps:
        # Expected shape: [B, H, N_img, N_txt]. For backward compatibility, also
        # accept full [B,H,N,N] maps with N_img_tokens provided.
        if N_img_tokens is not None and attn_block.shape[-1] == attn_block.shape[-2]:
            text_attn = attn_block[:, :, :N_img_tokens, N_img_tokens:]
        else:
            text_attn = attn_block
        entropy = -(text_attn * text_attn.clamp_min(1e-8).log()).sum(dim=-1).mean()
        total_loss = entropy if total_loss is None else total_loss + entropy
        count += 1

    if count == 0:
        return _zero_like_device(attn_maps)
    return total_loss / count


def _normalize_feature(x, eps=1e-6):
    x = x.float()
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def feature_divergence_loss(feat_adv_list, feat_clean_list):
    """Loss B: image-stream representation divergence.

    Larger value means adversarial image-stream features are farther from clean features.
    """
    total_loss = None
    count = 0
    for feat_adv, feat_clean in zip(feat_adv_list, feat_clean_list):
        if feat_adv is None or feat_clean is None:
            continue
        adv = _normalize_feature(feat_adv)
        clean = _normalize_feature(feat_clean).detach()

        cos_sim = F.cosine_similarity(adv.flatten(1), clean.flatten(1), dim=1)
        cos_dist = 1.0 - cos_sim.mean()

        gram_adv = _gram_matrix(adv)
        gram_clean = _gram_matrix(clean)
        style_dist = F.l1_loss(gram_adv, gram_clean)

        loss = cos_dist + style_dist
        total_loss = loss if total_loss is None else total_loss + loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device='cpu')
    return total_loss / count


def _gram_matrix(x):
    """Compute channel Gram matrix for [B, N, D] features."""
    B, N, D = x.shape
    feat = x.reshape(B, N, D)
    return torch.bmm(feat.transpose(1, 2), feat) / max(N * D, 1)


def trajectory_divergence_loss(v_adv, v_clean):
    """Loss C: flow trajectory direction divergence.

    Larger value means velocity predictions point in more different directions.
    """
    cos_sim = F.cosine_similarity(v_adv.float().flatten(1), v_clean.float().detach().flatten(1), dim=1)
    return 1.0 - cos_sim.mean()


def modality_imbalance_loss(img_stream_feats, txt_stream_feats):
    """Loss D: modality balance disruption.

    Larger value encourages larger image-stream variance and lower image/text correlation.
    This remains a proxy and must be interpreted with mechanism diagnostics.
    """
    total_loss = None
    count = 0

    for img_feat, txt_feat in zip(img_stream_feats, txt_stream_feats):
        if img_feat is None or txt_feat is None:
            continue
        img = img_feat.float()
        txt = txt_feat.float()

        img_var = img.var(dim=[1, 2], unbiased=False).mean()
        cross_corr = torch.bmm(
            F.normalize(img, dim=-1),
            F.normalize(txt, dim=-1).transpose(1, 2),
        )
        corr_abs = cross_corr.abs().mean()
        loss = img_var + 0.1 * (1.0 - corr_abs)

        total_loss = loss if total_loss is None else total_loss + loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device='cpu')
    return total_loss / count


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

        loss_history = {'total': [], 'textual': [], 'mmdit': []}
        pbar = tqdm(range(self.iters), desc=f"SD3-PGD mode={self.mmdit_mode}")

        feat_clean_list = None
        if self.mmdit_mode == 'B':
            feat_clean_list = self._compute_clean_features(X)

        for i in pbar:
            X_adv.requires_grad_(True)
            loss, components = self._compute_loss(X_adv, X, target_image, feat_clean_list, device)

            pbar.set_description(
                f"SD3-PGD mode={self.mmdit_mode} | total={loss.item():.3f} "
                f"textual={components['textual']:.3f} mmdit={components['mmdit']:.3f}"
            )

            if self.debug_grad and (i == 0 or i == self.iters - 1):
                self._print_grad_debug(X_adv, loss, components, i)

            loss.backward()
            grad = X_adv.grad.detach()

            X_adv = X_adv.detach() + self.g_dir * grad.sign() * self.step_size
            X_adv = torch.minimum(torch.maximum(X_adv, X - self.eps), X + self.eps)
            X_adv = torch.clamp(X_adv, min=self.clip_min, max=self.clip_max)

            loss_history['total'].append(loss.item())
            loss_history['textual'].append(components['textual'])
            loss_history['mmdit'].append(components['mmdit'])

            torch.cuda.empty_cache()

        return X_adv, loss_history

    def _grad_norm(self, tensor, X_adv, retain_graph=True):
        if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
            return 0.0
        grad = torch.autograd.grad(tensor, X_adv, retain_graph=retain_graph, allow_unused=True)[0]
        if grad is None:
            return 0.0
        return grad.detach().float().norm().item()

    def _print_grad_debug(self, X_adv, loss, components, step):
        text_tensor = components.get('_textual_tensor')
        mmdit_tensor = components.get('_mmdit_tensor')
        text_norm = self._grad_norm(text_tensor, X_adv, retain_graph=True)
        mmdit_norm = self._grad_norm(mmdit_tensor, X_adv, retain_graph=True)
        total_norm = self._grad_norm(loss, X_adv, retain_graph=True)
        print(
            f"[debug_grad step={step}] "
            f"textual={text_norm:.6e} mmdit={mmdit_norm:.6e} total={total_norm:.6e}"
        )

    def _compute_clean_features(self, X_clean):
        pipe = self.net.pipe
        hook = AttentionMapHook()
        transformer = pipe.transformer

        register_feature_hooks(
            transformer, hook, capture_attn=False, detach_features=True,
            capture_blocks=self.capture_blocks,
        )

        with torch.no_grad():
            z_clean = pipe.vae.encode(X_clean.to(pipe.vae.dtype)).latent_dist.mean
            z_clean = (z_clean - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            z_clean = z_clean.to(pipe.transformer.dtype)

            timestep = torch.tensor([0.5], device=X_clean.device, dtype=pipe.transformer.dtype)
            prompt_embeds = self.net.prompt_embeds.to(pipe.transformer.dtype)
            pooled_embeds = self.net.pooled_prompt_embeds.to(pipe.transformer.dtype)

            _ = transformer(
                hidden_states=z_clean,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )

        feat_clean = [f.clone().detach() for f in hook.img_stream_feats]
        hook.remove()
        restore_processors(transformer)
        return feat_clean

    def _textual_loss(self, z_adv, target_image, pipe, device):
        textual_loss = torch.tensor(0.0, device=device, dtype=z_adv.dtype)
        if target_image is None:
            return textual_loss
        with torch.no_grad():
            z_target = pipe.vae.encode(target_image.to(pipe.vae.dtype)).latent_dist.mean
            z_target = (z_target - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            z_target = z_target.to(z_adv.dtype).detach()
        raw_mse = self.cirt(z_adv, z_target)
        if self.textual_objective == 'toward_target':
            return -raw_mse
        if self.textual_objective == 'away_from_target':
            return raw_mse
        raise ValueError(f"Unknown textual_objective: {self.textual_objective}")

    def _compute_loss(self, X_adv, X_clean, target_image, feat_clean_list, device):
        pipe = self.net.pipe
        transformer = pipe.transformer

        z_adv = pipe.vae.encode(X_adv.to(pipe.vae.dtype)).latent_dist.mean
        z_adv = (z_adv - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        z_adv = z_adv.to(pipe.transformer.dtype)

        timestep = torch.tensor([torch.rand(1, device=device).item()], device=device, dtype=pipe.transformer.dtype)
        prompt_embeds = self.net.prompt_embeds.to(pipe.transformer.dtype)
        pooled_embeds = self.net.pooled_prompt_embeds.to(pipe.transformer.dtype)

        textual_loss = self._textual_loss(z_adv, target_image, pipe, device)
        mmdit_loss = torch.tensor(0.0, device=device, dtype=pipe.transformer.dtype)

        if self.mmdit_mode in {'O', 'O_repo'}:
            v_pred = transformer(
                hidden_states=z_adv,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
            mmdit_loss = denoiser_prediction_loss(v_pred)
        else:
            hook = AttentionMapHook()
            need_attn = (self.mmdit_mode == 'A')
            register_feature_hooks(
                transformer, hook, capture_attn=need_attn, detach_features=False,
                capture_blocks=self.capture_blocks,
            )

            v_pred = transformer(
                hidden_states=z_adv,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

            if self.mmdit_mode == 'A':
                mmdit_loss = cross_modal_disruption_loss(hook.attn_maps)
            elif self.mmdit_mode == 'B':
                if len(hook.img_stream_feats) > 0 and feat_clean_list is not None:
                    mmdit_loss = feature_divergence_loss(hook.img_stream_feats, feat_clean_list)
            elif self.mmdit_mode == 'C':
                with torch.no_grad():
                    z_clean = pipe.vae.encode(X_clean.to(pipe.vae.dtype)).latent_dist.mean
                    z_clean = (z_clean - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                    z_clean = z_clean.to(pipe.transformer.dtype)
                    v_clean = transformer(
                        hidden_states=z_clean,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds,
                        return_dict=False,
                    )[0]
                mmdit_loss = trajectory_divergence_loss(v_pred, v_clean)
            elif self.mmdit_mode == 'D':
                if len(hook.img_stream_feats) > 0 and len(hook.txt_stream_feats) > 0:
                    mmdit_loss = modality_imbalance_loss(hook.img_stream_feats, hook.txt_stream_feats)
            elif self.mmdit_mode == 'O_fair':
                with torch.no_grad():
                    z_clean = pipe.vae.encode(X_clean.to(pipe.vae.dtype)).latent_dist.mean
                    z_clean = (z_clean - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                    z_clean = z_clean.to(pipe.transformer.dtype)
                    v_clean = transformer(
                        hidden_states=z_clean,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds,
                        return_dict=False,
                    )[0]
                mmdit_loss = trajectory_divergence_loss(v_pred, v_clean)
            else:
                raise ValueError(f"Unknown SD3 MMDiT attack mode: {self.mmdit_mode}")

            hook.remove()
            restore_processors(transformer)

        joint_loss = self.textual_weight * textual_loss + self.mmdit_weight * mmdit_loss
        components = {
            'textual': textual_loss.detach().float().item(),
            'mmdit': mmdit_loss.detach().float().item() if isinstance(mmdit_loss, torch.Tensor) else float(mmdit_loss),
            '_textual_tensor': textual_loss,
            '_mmdit_tensor': mmdit_loss,
        }
        return joint_loss, components
