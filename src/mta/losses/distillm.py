"""DistiLLM loss with adaptive SGO scheduler and replay buffer."""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from mta.losses.divergences import skew_kl, skew_reverse_kl

log = logging.getLogger(__name__)


@dataclass
class ReplayBuffer:
    """Replay buffer for student-generated outputs (SGO)."""

    max_size: int = 512
    buffer: list[dict[str, Any]] = field(default_factory=list)

    def add(self, items: list[dict[str, Any]]) -> None:
        self.buffer.extend(items)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size :]

    def sample(self, n: int) -> list[dict[str, Any]]:
        if not self.buffer:
            return []
        return random.sample(self.buffer, min(n, len(self.buffer)))

    def state_dict(self) -> dict:
        return {"buffer": self.buffer, "max_size": self.max_size}

    def load_state_dict(self, state: dict) -> None:
        self.buffer = state["buffer"]
        self.max_size = state["max_size"]


class DistillmLoss(nn.Module):
    """DistiLLM: Skew KL or Skew Reverse KL divergence loss."""

    def __init__(
        self,
        loss_type: str = "srkl",
        alpha: float = 0.1,
        chunk_size: int = 512,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.chunk_size = chunk_size

    def forward(
        self,
        t_logits: torch.Tensor,
        s_logits: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_type == "skl":
            return skew_kl(t_logits, s_logits, response_mask, self.alpha, self.chunk_size)
        elif self.loss_type == "srkl":
            return skew_reverse_kl(t_logits, s_logits, response_mask, self.alpha, self.chunk_size)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
