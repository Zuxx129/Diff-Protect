#!/usr/bin/env python3
"""V2 wrappers for SD3 attack losses.

This module preserves the main A/B/C/D implementation in attacks_SD3.py and
only overrides O_fair so it is no longer identical to Mode C. In V2.1:

- Mode C: shared-noise latent trajectory divergence.
- O_fair: raw clean/adv latent velocity divergence without shared-noise trajectory.
"""
from __future__ import annotations

import torch

from attacks_SD3 import SD3_Linf_PGD as _BaseSD3_Linf_PGD
from attacks_SD3 import trajectory_divergence_loss


class SD3_Linf_PGD(_BaseSD3_Linf_PGD):
    """V2.1 PGD wrapper that defines O_fair as an independent baseline."""

    def _raw_velocity_baseline_loss(self, z_adv, X_clean, timestep, prompt_embeds, pooled_embeds):
        pipe = self.net.pipe
        transformer = pipe.transformer
        with torch.no_grad():
            z_clean = self._encode_latent(X_clean, pipe).detach()
            v_clean = transformer(
                hidden_states=z_clean,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
        v_adv = transformer(
            hidden_states=z_adv,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        loss, diagnostics = trajectory_divergence_loss(v_adv, v_clean, return_diagnostics=True)
        diagnostics["ofair_raw_velocity_div"] = diagnostics.get("velocity_div", 0.0)
        return loss, diagnostics

    def _compute_loss(self, X_adv, X_clean, target_image, device):
        if self.mmdit_mode != "O_fair":
            return super()._compute_loss(X_adv, X_clean, target_image, device)

        pipe = self.net.pipe
        z_adv = self._encode_latent(X_adv, pipe)
        timestep = torch.tensor([torch.rand(1, device=device).item()], device=device, dtype=pipe.transformer.dtype)
        prompt_embeds = self.net.prompt_embeds.to(pipe.transformer.dtype)
        pooled_embeds = self.net.pooled_prompt_embeds.to(pipe.transformer.dtype)

        textual_loss = self._textual_loss(z_adv, target_image, pipe, device)
        mmdit_loss, diagnostics = self._raw_velocity_baseline_loss(
            z_adv, X_clean, timestep, prompt_embeds, pooled_embeds
        )

        joint_loss = self.textual_weight * textual_loss + self.mmdit_weight * mmdit_loss
        components = {
            "textual": textual_loss.detach().float().item(),
            "mmdit": mmdit_loss.detach().float().item(),
            "_textual_tensor": textual_loss,
            "_mmdit_tensor": mmdit_loss,
        }
        components.update(diagnostics)
        return joint_loss, components
