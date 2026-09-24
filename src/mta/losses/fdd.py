"""Feature Dynamics Distillation loss (Eq. 3-4)."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def _logit_lens_chunk(
        self,
        hidden_chunk: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        """Apply logit lens on a chunk: norm -> lm_head -> log_softmax."""
        h = hidden_chunk
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
        B, T = response_mask.shape
        cs = self.chunk_size if self.chunk_size > 0 else T
        n_tokens = response_mask.sum(dim=1).float().clamp(min=1)

        traj_accum = torch.zeros(B, device=device)
        der_accum = torch.zeros(B, device=device)

        for start in range(0, T, cs):
            end = min(start + cs, T)
            mask_c = response_mask[:, start:end].float()

            prev_s_chunk = None
            prev_t_chunk = None

            for j, l_s in enumerate(self.layers):
                l_t = math.floor(l_s * self.n_teacher / self.n_student)

                s_chunk = self._logit_lens_chunk(s_hidden[l_s][:, start:end, :], s_model)
                with torch.no_grad():
                    t_chunk = self._logit_lens_chunk(t_hidden[l_t][:, start:end, :], t_model)

                # L_Traj: KL(teacher || student)
                t_prob = t_chunk.exp()
                kl = (t_prob * (t_chunk - s_chunk)).sum(dim=-1)
                traj_accum = traj_accum + (kl * mask_c).sum(dim=1)

                # L_Der: cosine between layer derivatives
                if prev_s_chunk is not None:
                    delta_s = s_chunk - prev_s_chunk
                    with torch.no_grad():
                        delta_t = t_chunk - prev_t_chunk
                    cos = F.cosine_similarity(
                        delta_s.reshape(-1, delta_s.shape[-1]),
                        delta_t.reshape(-1, delta_t.shape[-1]),
                        dim=-1,
                    )
                    mask_flat = mask_c.reshape(-1)
                    der_accum = der_accum + ((1.0 - cos) * mask_flat).sum().unsqueeze(0).expand(B)

                prev_s_chunk = s_chunk.detach()
                prev_t_chunk = t_chunk

            del prev_s_chunk, prev_t_chunk

        traj_loss = (traj_accum / n_tokens).mean()
        n_mask_total = response_mask.sum().float().clamp(min=1)
        der_loss = der_accum.sum() / (B * n_mask_total) if der_accum.sum() > 0 else torch.tensor(0.0, device=device)

        results = {
            "traj": traj_loss,
            "der": der_loss,
        }
        total = self.lambda_traj * traj_loss + self.lambda_der * der_loss

        if self.include_final_kd:
            kd_accum = torch.zeros(B, device=device)
            for start in range(0, T, cs):
                end = min(start + cs, T)
                mask_c = response_mask[:, start:end].float()
                s_chunk = self._logit_lens_chunk(s_hidden[-1][:, start:end, :], s_model)
                with torch.no_grad():
                    t_chunk = self._logit_lens_chunk(t_hidden[-1][:, start:end, :], t_model)
                t_prob = t_chunk.exp()
                kl = (t_prob * (t_chunk - s_chunk)).sum(dim=-1)
                kd_accum = kd_accum + (kl * mask_c).sum(dim=1)
            kd = (kd_accum / n_tokens).mean()
            results["kd"] = kd
            total = total + self.lambda_kd * kd

        results["total"] = total
        return results
