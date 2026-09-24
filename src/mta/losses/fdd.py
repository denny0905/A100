"""Feature Dynamics Distillation loss (Eq. 3-4)."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mta.losses.divergences import kl_divergence

log = logging.getLogger(__name__)


class FDDLoss(nn.Module):
    """FDD: trajectory KL + derivative cosine over intermediate layers."""

    def __init__(
        self,
        layers: list[int],
        n_student: int,
        n_teacher: int,
        apply_final_norm: bool = True,
        include_final_kd: bool = True,
        lambda_traj: float = 1.0,
        lambda_der: float = 1.0,
        lambda_kd: float = 1.0,
        chunk_size: int = 512,
    ) -> None:
        super().__init__()
        self.layers = sorted(layers)
        self.n_student = n_student
        self.n_teacher = n_teacher
        self.apply_final_norm = apply_final_norm
        self.include_final_kd = include_final_kd
        self.lambda_traj = lambda_traj
        self.lambda_der = lambda_der
        self.lambda_kd = lambda_kd
        self.chunk_size = chunk_size

    def _logit_lens(
        self,
        hidden: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        """Apply logit lens: optional final_norm then lm_head, then log_softmax."""
        h = hidden
        if self.apply_final_norm and hasattr(model, "transformer"):
            h = model.transformer.ln_f(h.to(model.transformer.ln_f.weight.dtype))
        elif self.apply_final_norm and hasattr(model, "model"):
            if hasattr(model.model, "norm"):
                h = model.model.norm(h.to(model.model.norm.weight.dtype))
            elif hasattr(model.model, "final_layernorm"):
                h = model.model.final_layernorm(h.to(model.model.final_layernorm.weight.dtype))
        if hasattr(model, "lm_head"):
            logits = model.lm_head(h.to(model.lm_head.weight.dtype))
        else:
            logits = h
        return F.log_softmax(logits.float(), dim=-1)

    def forward(
        self,
        s_hidden: tuple[torch.Tensor, ...],
        t_hidden: tuple[torch.Tensor, ...],
        s_model: nn.Module,
        t_model: nn.Module,
        response_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = response_mask.device
        traj_loss = torch.tensor(0.0, device=device)
        der_loss = torch.tensor(0.0, device=device)

        prev_s_logits = None
        prev_t_logits = None

        for j, l_s in enumerate(self.layers):
            l_t = math.floor(l_s * self.n_teacher / self.n_student)
            with torch.no_grad():
                t_logits = self._logit_lens(t_hidden[l_t], t_model)
            s_logits = self._logit_lens(s_hidden[l_s], s_model)

            # L_Traj: KL(teacher || student) at this layer
            traj_loss = traj_loss + kl_divergence(
                t_logits, s_logits, response_mask, self.chunk_size
            )

            # L_Der: cosine between derivatives
            if prev_s_logits is not None:
                delta_s = s_logits - prev_s_logits
                with torch.no_grad():
                    delta_t = t_logits - prev_t_logits
                cos = F.cosine_similarity(
                    delta_s.reshape(-1, delta_s.shape[-1]),
                    delta_t.reshape(-1, delta_t.shape[-1]),
                    dim=-1,
                )
                mask_flat = response_mask.reshape(-1).float()
                n = mask_flat.sum().clamp(min=1)
                der_loss = der_loss + ((1.0 - cos) * mask_flat).sum() / n

            prev_s_logits = s_logits.detach()
            prev_t_logits = t_logits

        results = {
            "traj": traj_loss,
            "der": der_loss,
        }

        total = self.lambda_traj * traj_loss + self.lambda_der * der_loss

        if self.include_final_kd:
            with torch.no_grad():
                t_final_logits = self._logit_lens(t_hidden[-1], t_model)
            s_final_logits = self._logit_lens(s_hidden[-1], s_model)
            kd = kl_divergence(
                t_final_logits,
                s_final_logits,
                response_mask,
                self.chunk_size,
            )
            results["kd"] = kd
            total = total + self.lambda_kd * kd

        results["total"] = total
        return results
