#!/usr/bin/env python3
"""SD3 v3 attack objectives.

This module keeps the v3 experiment semantics separate from the v2.1 joint
objective implementation.  The public attack class is `SD3V3LinfPGD`.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from attacks_SD3 import (
    AttentionMapHook,
    cross_modal_disruption_loss,
    denoiser_prediction_loss,
    feature_divergence_loss,
    modality_imbalance_loss,
    register_feature_hooks,
    restore_processors,
    trajectory_divergence_loss,
)


V3_MODES = {
    "textual_only",
    "E",
    "O",
    "A",
    "B",
    "C",
    "D",
    "FMP_single",
    "FMP_multi",
    "FMP_single_plus_step",
}


def _tensor_item(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)


def _as_float_list(values: Optional[Iterable[float]]) -> List[float]:
    if values is None:
        return [0.1, 0.3, 0.5]
    return [float(v) for v in values]


class SD3V3LinfPGD:
    """L_inf PGD for the SD3 v3 method set.

    Objective convention: every mode returns a scalar where a larger value means
    stronger attack pressure.  `opt_direction="maximize"` therefore uses
    gradient ascent; `opt_direction="minimize"` is kept only for diagnostics.
    """

    def __init__(
        self,
        net,
        fn,
        epsilon: float,
        steps: int,
        eps_iter: float,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        opt_direction: str = "maximize",
        mode: str = "FMP_single",
        textual_weight: float = 1.0,
        mmdit_weight: float = 100000.0,
        fmp_sigma_levels: Optional[Iterable[float]] = None,
        fmp_multi_reduce: str = "mean",
        fmp_target_convention: str = "noise_minus_data",
        fmp_loss_eta: float = 1e-8,
        use_step_loss: bool = False,
        lambda_step: float = 1.0,
        sdedit_steps: int = 28,
        debug_grad: bool = False,
        capture_blocks=None,
    ):
        if mode not in V3_MODES:
            raise ValueError(f"Unknown SD3 v3 attack mode: {mode}")
        if fmp_multi_reduce != "mean":
            raise ValueError("v3 currently supports only fmp_multi_reduce=mean")
        if fmp_target_convention not in {"noise_minus_data", "data_minus_noise"}:
            raise ValueError("fmp_target_convention must be noise_minus_data or data_minus_noise")

        self.net = net
        self.fn = fn
        self.eps = float(epsilon)
        self.step_size = float(eps_iter)
        self.iters = int(steps)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.opt_direction = opt_direction
        self.g_dir = 1.0 if opt_direction == "maximize" else -1.0
        self.mode = mode
        self.textual_weight = float(textual_weight)
        self.mmdit_weight = float(mmdit_weight)
        self.fmp_sigma_levels = _as_float_list(fmp_sigma_levels)
        self.fmp_multi_reduce = fmp_multi_reduce
        self.fmp_target_convention = fmp_target_convention
        self.fmp_loss_eta = float(fmp_loss_eta)
        self.use_step_loss = bool(use_step_loss or mode == "FMP_single_plus_step")
        self.lambda_step = float(lambda_step)
        self.sdedit_steps = int(sdedit_steps)
        self.debug_grad = bool(debug_grad)
        self.capture_blocks = capture_blocks
        self.cirt = nn.MSELoss(reduction="sum")

        if self.use_step_loss and mode not in {"FMP_single", "FMP_single_plus_step"}:
            raise ValueError("use_step_loss is only defined for FMP_single/FMP_single_plus_step")

    def pgd_sd3(self, X, target_image=None, random_start: bool = False):
        device = X.device
        if random_start:
            X_adv = X.detach() + torch.empty_like(X).uniform_(-self.eps, self.eps)
            X_adv = torch.minimum(torch.maximum(X_adv, X - self.eps), X + self.eps)
            X_adv = torch.clamp(X_adv, min=self.clip_min, max=self.clip_max)
        else:
            X_adv = X.detach().clone()

        loss_history: Dict[str, List[float]] = {}
        pbar = tqdm(range(self.iters), desc=f"SD3-v3-PGD mode={self.mode}")

        for step in pbar:
            X_adv.requires_grad_(True)
            loss, components = self._compute_loss(X_adv, X, target_image, device)

            if self.debug_grad:
                components.update(self._grad_debug_dict(X_adv, loss, components))

            pbar.set_description(
                f"SD3-v3-PGD mode={self.mode} | total={loss.item():.4f}"
            )
            if self.debug_grad and (step == 0 or step == self.iters - 1):
                print(
                    f"[v3 debug_grad step={step}] "
                    f"total={components.get('grad_total_l2', 0.0):.6e} "
                    f"fmp={components.get('grad_fmp_l2', 0.0):.6e} "
                    f"step={components.get('grad_step_l2', 0.0):.6e} "
                    f"o={components.get('grad_o_l2', 0.0):.6e} "
                    f"textual={components.get('grad_textual_l2', 0.0):.6e}"
                )

            if not self.debug_grad:
                if "_loss_fmp_tensor" in components:
                    components["grad_fmp_l2"] = self._grad_norm(
                        components.get("_loss_fmp_tensor"), X_adv, retain_graph=True
                    )
                if "_loss_step_tensor" in components:
                    components["grad_step_l2"] = self._grad_norm(
                        components.get("_loss_step_tensor"), X_adv, retain_graph=True
                    )

            loss.backward()
            grad = X_adv.grad.detach() if X_adv.grad is not None else torch.zeros_like(X_adv)
            components.setdefault("grad_total_l2", float(grad.detach().float().norm().item()))
            X_adv = X_adv.detach() + self.g_dir * grad.sign() * self.step_size
            X_adv = torch.minimum(torch.maximum(X_adv, X - self.eps), X + self.eps)
            X_adv = torch.clamp(X_adv, min=self.clip_min, max=self.clip_max)

            record = {"total": _tensor_item(loss), "loss_total": _tensor_item(loss)}
            record["linf"] = float(torch.abs(X_adv.detach() - X.detach()).max().item())
            for key, val in components.items():
                if key.startswith("_"):
                    continue
                record[key] = _tensor_item(val)
            for key, val in record.items():
                loss_history.setdefault(key, []).append(val)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return X_adv, loss_history

    def _grad_norm(self, tensor, X_adv, retain_graph=True) -> float:
        if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
            return 0.0
        grad = torch.autograd.grad(tensor, X_adv, retain_graph=retain_graph, allow_unused=True)[0]
        if grad is None:
            return 0.0
        return float(grad.detach().float().norm().item())

    def _grad_debug_dict(self, X_adv, loss, components) -> Dict[str, float]:
        return {
            "grad_total_l2": self._grad_norm(loss, X_adv, retain_graph=True),
            "grad_fmp_l2": self._grad_norm(components.get("_loss_fmp_tensor"), X_adv, retain_graph=True),
            "grad_step_l2": self._grad_norm(components.get("_loss_step_tensor"), X_adv, retain_graph=True),
            "grad_o_l2": self._grad_norm(components.get("_loss_o_tensor"), X_adv, retain_graph=True),
            "grad_textual_l2": self._grad_norm(components.get("_loss_textual_mse_tensor"), X_adv, retain_graph=True),
            "grad_mechanism_l2": self._grad_norm(components.get("_loss_mechanism_tensor"), X_adv, retain_graph=True),
        }

    def _encode_latent(self, image, pipe):
        z = pipe.vae.encode(image.to(pipe.vae.dtype)).latent_dist.mean
        z = (z - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        return z.to(pipe.transformer.dtype)

    def _prompt_tensors(self, transformer):
        return (
            self.net.prompt_embeds.to(transformer.dtype),
            self.net.pooled_prompt_embeds.to(transformer.dtype),
        )

    def _scheduler_match(self, sigma_value: float, device, dtype):
        """Map an SDEdit sigma level to the closest SD3 scheduler state."""
        pipe = self.net.pipe
        pipe.scheduler.set_timesteps(self.sdedit_steps, device=device)
        sigmas = pipe.scheduler.sigmas.to(device=device)
        idx = int((sigmas.float() - float(sigma_value)).abs().argmin().item())
        idx = min(idx, len(pipe.scheduler.timesteps) - 1)
        timestep = pipe.scheduler.timesteps[idx].to(device=device)
        timestep = timestep.reshape(1).to(dtype=dtype)
        return timestep, float(timestep.detach().float().item()), idx, sigmas

    def _sigma_to_timestep(self, sigma_value: float, device, dtype) -> Tuple[torch.Tensor, float]:
        """Map an SDEdit sigma level to the closest SD3 scheduler timestep."""
        timestep, timestep_value, _, _ = self._scheduler_match(sigma_value, device, dtype)
        return timestep, timestep_value

    def _step_h_for_sigma(self, sigma_value: float, device, dtype) -> float:
        """Approximate one Euler step length around the matched scheduler sigma."""
        _, _, idx, sigmas = self._scheduler_match(sigma_value, device, dtype)
        if len(sigmas) <= 1:
            return 1.0 / max(self.sdedit_steps, 1)
        if idx < len(sigmas) - 1:
            h = (sigmas[idx] - sigmas[idx + 1]).abs()
        else:
            h = (sigmas[idx - 1] - sigmas[idx]).abs()
        return float(h.detach().float().item())

    def _sample_sigma_value(self, device) -> float:
        idx = int(torch.randint(0, len(self.fmp_sigma_levels), (1,), device=device).item())
        return float(self.fmp_sigma_levels[idx])

    def _call_transformer(self, z, timestep, prompt_embeds, pooled_embeds):
        transformer = self.net.pipe.transformer
        if timestep.ndim == 0:
            timestep = timestep.reshape(1)
        if timestep.shape[0] != z.shape[0]:
            timestep = timestep.expand(z.shape[0])
        return transformer(
            hidden_states=z,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    def _noised_latent(self, z, sigma_value: float, noise=None):
        sigma = torch.tensor(float(sigma_value), device=z.device, dtype=z.dtype).reshape(1, 1, 1, 1)
        if noise is None:
            noise = torch.randn_like(z)
        return (1.0 - sigma) * z + sigma * noise, noise

    def _textual_mse(self, z_adv, target_image, pipe, device):
        if target_image is None:
            return z_adv.float().sum().to(z_adv.dtype) * 0.0
        with torch.no_grad():
            z_target = self._encode_latent(target_image, pipe).to(z_adv.dtype).detach()
        return self.cirt(z_adv, z_target)

    def _semantic_o_terms(
        self,
        z_adv,
        prompt_embeds,
        pooled_embeds,
        device,
        sigma_value=None,
        noise=None,
        use_noised_state: bool = False,
    ):
        sigma_value = self._sample_sigma_value(device) if sigma_value is None else float(sigma_value)
        timestep, timestep_value = self._sigma_to_timestep(sigma_value, device, self.net.pipe.transformer.dtype)
        z_input = z_adv
        if use_noised_state:
            z_input, noise = self._noised_latent(z_adv, sigma_value, noise=noise)
        v_pred = self._call_transformer(z_input, timestep, prompt_embeds, pooled_embeds)
        loss_o = denoiser_prediction_loss(v_pred)
        diagnostics = {
            "loss_o": loss_o,
            "sampled_sigma": sigma_value,
            "sampled_timestep": timestep_value,
            "v_pred_norm": v_pred.detach().float().norm(),
            "_loss_o_tensor": loss_o,
        }
        return loss_o, diagnostics

    def _fmp_terms(self, z_adv, prompt_embeds, pooled_embeds, device, sigma_value=None, noise=None):
        sigma_value = self._sample_sigma_value(device) if sigma_value is None else float(sigma_value)
        timestep, timestep_value = self._sigma_to_timestep(sigma_value, device, self.net.pipe.transformer.dtype)
        z_sigma, noise = self._noised_latent(z_adv, sigma_value, noise=noise)
        if self.fmp_target_convention == "noise_minus_data":
            u_sigma = noise - z_adv
        else:
            u_sigma = z_adv - noise
        u_sigma = u_sigma.detach()
        v_pred = self._call_transformer(z_sigma, timestep, prompt_embeds, pooled_embeds)
        diff = v_pred.float() - u_sigma.float()
        pred_error = diff.pow(2).sum()
        u_norm_sq = u_sigma.float().pow(2).sum().detach()
        loss_fmp = (pred_error / (u_norm_sq + self.fmp_loss_eta)).to(v_pred.dtype)
        diagnostics = {
            "loss_fmp": loss_fmp,
            "sampled_sigma": sigma_value,
            "sampled_timestep": timestep_value,
            "u_norm": u_sigma.detach().float().norm(),
            "v_pred_norm": v_pred.detach().float().norm(),
            "prediction_error": pred_error.detach(),
            "_loss_fmp_tensor": loss_fmp,
            "_z_sigma_tensor": z_sigma,
            "_noise_tensor": noise,
            "_v_pred_tensor": v_pred,
            "_timestep_tensor": timestep,
        }
        return loss_fmp, diagnostics

    def _fmp_multi_terms(self, z_adv, prompt_embeds, pooled_embeds, device):
        losses = []
        diag_values: Dict[str, List[float]] = {}
        for sigma_value in self.fmp_sigma_levels:
            loss_i, diag_i = self._fmp_terms(
                z_adv, prompt_embeds, pooled_embeds, device, sigma_value=sigma_value
            )
            losses.append(loss_i)
            for key, value in diag_i.items():
                if key.startswith("_"):
                    continue
                diag_values.setdefault(key, []).append(_tensor_item(value))
        loss = torch.stack(losses).mean()
        diagnostics = {key: sum(vals) / len(vals) for key, vals in diag_values.items()}
        diagnostics["loss_fmp"] = loss
        diagnostics["num_sigma_samples"] = float(len(losses))
        diagnostics["_loss_fmp_tensor"] = loss
        return loss, diagnostics

    def _fmp_single_plus_step_terms(self, z_adv, z_clean, prompt_embeds, pooled_embeds, device):
        sigma_value = self._sample_sigma_value(device)
        noise = torch.randn_like(z_adv).detach()
        loss_fmp, diagnostics = self._fmp_terms(
            z_adv, prompt_embeds, pooled_embeds, device, sigma_value=sigma_value, noise=noise
        )

        timestep = diagnostics["_timestep_tensor"]
        z_sigma_adv = diagnostics["_z_sigma_tensor"]
        v_adv = diagnostics["_v_pred_tensor"]
        step_h = self._step_h_for_sigma(sigma_value, device, self.net.pipe.transformer.dtype)
        h_tensor = torch.tensor(step_h, device=z_adv.device, dtype=z_adv.dtype).reshape(1, 1, 1, 1)
        sigma = torch.tensor(float(sigma_value), device=z_adv.device, dtype=z_adv.dtype).reshape(1, 1, 1, 1)

        with torch.no_grad():
            z_sigma_clean = (1.0 - sigma) * z_clean.detach() + sigma * noise
            v_clean = self._call_transformer(z_sigma_clean, timestep, prompt_embeds, pooled_embeds).detach()
            transition_ref = (z_sigma_clean + h_tensor * v_clean).detach()

        transition_adv = z_sigma_adv + h_tensor * v_adv
        transition_sep = (transition_adv.float() - transition_ref.float()).pow(2).sum()
        ref_norm_sq = transition_ref.float().pow(2).sum().detach()
        loss_step = (transition_sep / (ref_norm_sq + self.fmp_loss_eta)).to(v_adv.dtype)
        loss = loss_fmp + self.lambda_step * loss_step

        diagnostics.update(
            {
                "loss_step": loss_step,
                "lambda_step": self.lambda_step,
                "transition_sep": transition_sep.detach(),
                "step_h": step_h,
                "_loss_step_tensor": loss_step,
            }
        )
        return loss, diagnostics

    def _compute_clean_features_at_timestep(self, z_clean, timestep, prompt_embeds, pooled_embeds, capture_attn=False):
        pipe = self.net.pipe
        transformer = pipe.transformer
        hook = AttentionMapHook()
        register_feature_hooks(
            transformer,
            hook,
            capture_attn=capture_attn,
            detach_features=True,
            capture_blocks=self.capture_blocks,
        )
        try:
            with torch.no_grad():
                _ = self._call_transformer(z_clean.detach(), timestep, prompt_embeds, pooled_embeds)
            return {
                "img": [f.clone().detach() for f in hook.img_stream_feats],
                "txt": [f.clone().detach() if f is not None else None for f in hook.txt_stream_feats],
                "attn": [f.clone().detach() for f in hook.attn_maps],
                "injection": [f.clone().detach() for f in hook.txt_injection_feats],
            }
        finally:
            hook.remove()
            restore_processors(transformer)

    def _trajectory_loss_shared_noise(self, z_adv, z_clean, timestep, prompt_embeds, pooled_embeds, sigma_value):
        noise = torch.randn_like(z_adv).detach()
        sigma = torch.tensor(float(sigma_value), device=z_adv.device, dtype=z_adv.dtype).reshape(1, 1, 1, 1)
        z_clean_t = (1.0 - sigma) * z_clean.detach() + sigma * noise
        with torch.no_grad():
            v_clean = self._call_transformer(z_clean_t, timestep, prompt_embeds, pooled_embeds)
        z_adv_t = (1.0 - sigma) * z_adv + sigma * noise
        v_adv = self._call_transformer(z_adv_t, timestep, prompt_embeds, pooled_embeds)
        return trajectory_divergence_loss(v_adv, v_clean, return_diagnostics=True)

    def _mechanism_terms(self, mode, z_adv, z_clean, prompt_embeds, pooled_embeds, device):
        sigma_value = self._sample_sigma_value(device)
        timestep, timestep_value = self._sigma_to_timestep(sigma_value, device, self.net.pipe.transformer.dtype)
        diagnostics = {"sampled_sigma": sigma_value, "sampled_timestep": timestep_value}

        if mode == "C":
            loss, diag = self._trajectory_loss_shared_noise(
                z_adv, z_clean, timestep, prompt_embeds, pooled_embeds, sigma_value
            )
            diagnostics.update(diag)
            diagnostics["loss_mechanism"] = loss
            diagnostics["_loss_mechanism_tensor"] = loss
            return loss, diagnostics

        clean_ref = self._compute_clean_features_at_timestep(
            z_clean, timestep, prompt_embeds, pooled_embeds, capture_attn=(mode == "A")
        )
        hook = AttentionMapHook()
        transformer = self.net.pipe.transformer
        register_feature_hooks(
            transformer,
            hook,
            capture_attn=(mode == "A"),
            detach_features=False,
            capture_blocks=self.capture_blocks,
        )
        try:
            _ = self._call_transformer(z_adv, timestep, prompt_embeds, pooled_embeds)
            if mode == "A":
                loss, diag = cross_modal_disruption_loss(
                    hook.attn_maps,
                    injection_adv=hook.txt_injection_feats,
                    injection_clean=clean_ref["injection"],
                    return_diagnostics=True,
                )
            elif mode == "B":
                loss, diag = feature_divergence_loss(
                    hook.img_stream_feats, clean_ref["img"], return_diagnostics=True
                )
            elif mode == "D":
                loss, diag = modality_imbalance_loss(
                    hook.img_stream_feats,
                    hook.txt_stream_feats,
                    img_clean_feats=clean_ref["img"],
                    txt_clean_feats=clean_ref["txt"],
                    return_diagnostics=True,
                )
            else:
                raise ValueError(f"Unknown mechanism mode: {mode}")
        finally:
            hook.remove()
            restore_processors(transformer)

        diagnostics.update(diag)
        diagnostics["loss_mechanism"] = loss
        diagnostics["_loss_mechanism_tensor"] = loss
        return loss, diagnostics

    def _compute_loss(self, X_adv, X_clean, target_image, device):
        pipe = self.net.pipe
        transformer = pipe.transformer
        z_adv = self._encode_latent(X_adv, pipe)
        prompt_embeds, pooled_embeds = self._prompt_tensors(transformer)
        components: Dict[str, object] = {}

        if self.mode == "textual_only":
            textual_mse = self._textual_mse(z_adv, target_image, pipe, device)
            loss = -textual_mse
            components.update(
                {
                    "loss_textual_mse": textual_mse,
                    "textual_weight": self.textual_weight,
                    "_loss_textual_mse_tensor": textual_mse,
                }
            )
            return loss, components

        if self.mode == "E":
            loss, diag = self._semantic_o_terms(z_adv, prompt_embeds, pooled_embeds, device)
            components.update(diag)
            return loss, components

        if self.mode == "O":
            loss_o, diag = self._semantic_o_terms(z_adv, prompt_embeds, pooled_embeds, device)
            textual_mse = self._textual_mse(z_adv, target_image, pipe, device)
            loss = self.textual_weight * (-textual_mse) + self.mmdit_weight * loss_o
            components.update(diag)
            components.update(
                {
                    "loss_textual_mse": textual_mse,
                    "textual_weight": self.textual_weight,
                    "mmdit_weight": self.mmdit_weight,
                    "_loss_textual_mse_tensor": textual_mse,
                    "_loss_o_tensor": loss_o,
                }
            )
            return loss, components

        if self.mode == "FMP_single":
            if self.use_step_loss:
                with torch.no_grad():
                    z_clean = self._encode_latent(X_clean, pipe).detach()
                loss, diag = self._fmp_single_plus_step_terms(
                    z_adv, z_clean, prompt_embeds, pooled_embeds, device
                )
                components.update(diag)
                return loss, components
            loss, diag = self._fmp_terms(z_adv, prompt_embeds, pooled_embeds, device)
            components.update(diag)
            return loss, components

        if self.mode == "FMP_single_plus_step":
            with torch.no_grad():
                z_clean = self._encode_latent(X_clean, pipe).detach()
            loss, diag = self._fmp_single_plus_step_terms(
                z_adv, z_clean, prompt_embeds, pooled_embeds, device
            )
            components.update(diag)
            return loss, components

        if self.mode == "FMP_multi":
            loss, diag = self._fmp_multi_terms(z_adv, prompt_embeds, pooled_embeds, device)
            components.update(diag)
            return loss, components

        with torch.no_grad():
            z_clean = self._encode_latent(X_clean, pipe).detach()
        loss, diag = self._mechanism_terms(
            self.mode, z_adv, z_clean, prompt_embeds, pooled_embeds, device
        )
        textual_mse = self._textual_mse(z_adv, target_image, pipe, device)
        loss = self.textual_weight * (-textual_mse) + self.mmdit_weight * loss
        components.update(diag)
        components.update(
            {
                "loss_textual_mse": textual_mse,
                "textual_weight": self.textual_weight,
                "mmdit_weight": self.mmdit_weight,
                "_loss_textual_mse_tensor": textual_mse,
            }
        )
        return loss, components
