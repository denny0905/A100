"""DistiLLM-2 CALD loss with curriculum alpha/beta (Eq. 17-18)."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from mta.losses.divergences import skew_kl, skew_reverse_kl

log = logging.getLogger(__name__)


class Distillm2Loss(nn.Module):
    """CALD: Curriculum Adaptive Logit Distillation."""

    def __init__(
        self,
        alpha_init: float = 0.1,
        beta_init: float = 0.0,
        beta_final: float = 0.5,
        total_steps: int = 1000,
        chunk_size: int = 512,
    ) -> None:
        super().__init__()
        self.alpha_t = alpha_init
        self.alpha_s = alpha_init
        self.beta = beta_init
        self.beta_init = beta_init
        self.beta_final = beta_final
        self.total_steps = total_steps
        self.chunk_size = chunk_size
        self.step = 0

    def update_schedule(self, step: int) -> None:
        """Update beta linearly from beta_init to beta_final."""
        self.step = step
        progress = min(step / max(self.total_steps, 1), 1.0)
        self.beta = self.beta_init + (self.beta_final - self.beta_init) * progress

    def forward(
        self,
        t_logits_tgo: torch.Tensor,
        s_logits_tgo: torch.Tensor,
        t_logits_sgo: torch.Tensor,
        s_logits_sgo: torch.Tensor,
        response_mask_tgo: torch.Tensor,
        response_mask_sgo: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # SKL on TGO
        skl_tgo = skew_kl(
            t_logits_tgo, s_logits_tgo, response_mask_tgo,
            alpha=self.alpha_t, chunk_size=self.chunk_size,
        )
        # SRKL on SGO
        srkl_sgo = skew_reverse_kl(
            t_logits_sgo, s_logits_sgo, response_mask_sgo,
            alpha=self.alpha_s, chunk_size=self.chunk_size,
        )

        # Eq. 17
        loss = 0.5 * ((1 - self.beta) * skl_tgo + self.beta * srkl_sgo)

        return {
            "total": loss,
            "skl_tgo": skl_tgo.detach(),
            "srkl_sgo": srkl_sgo.detach(),
            "beta": torch.tensor(self.beta),
            "alpha_t": torch.tensor(self.alpha_t),
            "alpha_s": torch.tensor(self.alpha_s),
        }

    def state_dict_extra(self) -> dict:
        return {
            "alpha_t": self.alpha_t,
            "alpha_s": self.alpha_s,
            "beta": self.beta,
            "step": self.step,
        }

    def load_state_dict_extra(self, state: dict) -> None:
        self.alpha_t = state["alpha_t"]
        self.alpha_s = state["alpha_s"]
        self.beta = state["beta"]
        self.step = state["step"]
