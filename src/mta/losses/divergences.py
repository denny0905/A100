"""KL, Skew KL, Skew Reverse KL — chunked and masked for response tokens."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    response_mask: torch.Tensor,
    chunk_size: int = 0,
) -> torch.Tensor:
    """Forward KL: D_KL(p || q) on response tokens only.

    Args:
        p_logits: [B, T, V] teacher logits.
        q_logits: [B, T, V] student logits.
        response_mask: [B, T] 1 for response tokens.
        chunk_size: if >0, process in chunks along T to save memory.
    """
    if chunk_size > 0:
        return _chunked_kl(p_logits, q_logits, response_mask, chunk_size, "kl")
    return _kl_full(p_logits, q_logits, response_mask)


def _kl_full(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    p = F.log_softmax(p_logits.float(), dim=-1)
    q = F.log_softmax(q_logits.float(), dim=-1)
    p_prob = p.exp()
    # KL = sum_v p(v) * (log p(v) - log q(v))
    kl = (p_prob * (p - q)).sum(dim=-1)  # [B, T]
    kl = kl * response_mask.float()
    n_tokens = response_mask.sum(dim=1).clamp(min=1)
    return (kl.sum(dim=1) / n_tokens).mean()


def skew_kl(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float = 0.1,
    chunk_size: int = 0,
) -> torch.Tensor:
    """Skew KL: D_KL(p || alpha*p + (1-alpha)*q) (Eq. 15)."""
    if chunk_size > 0:
        return _chunked_kl(p_logits, q_logits, response_mask, chunk_size, "skl", alpha)
    return _skl_full(p_logits, q_logits, response_mask, alpha)


def _skl_full(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    p_log = F.log_softmax(p_logits.float(), dim=-1)
    q_log = F.log_softmax(q_logits.float(), dim=-1)
    p_prob = p_log.exp()
    q_prob = q_log.exp()
    mix = alpha * p_prob + (1 - alpha) * q_prob
    mix_log = mix.log()
    kl = (p_prob * (p_log - mix_log)).sum(dim=-1)  # [B, T]
    kl = kl * response_mask.float()
    n_tokens = response_mask.sum(dim=1).clamp(min=1)
    return (kl.sum(dim=1) / n_tokens).mean()


def skew_reverse_kl(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float = 0.1,
    chunk_size: int = 0,
) -> torch.Tensor:
    """Skew Reverse KL: D_KL(q || (1-alpha)*p + alpha*q) (Eq. 16)."""
    if chunk_size > 0:
        return _chunked_kl(p_logits, q_logits, response_mask, chunk_size, "srkl", alpha)
    return _srkl_full(p_logits, q_logits, response_mask, alpha)


def _srkl_full(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    p_log = F.log_softmax(p_logits.float(), dim=-1)
    q_log = F.log_softmax(q_logits.float(), dim=-1)
    p_prob = p_log.exp()
    q_prob = q_log.exp()
    mix = (1 - alpha) * p_prob + alpha * q_prob
    mix_log = mix.log()
    kl = (q_prob * (q_log - mix_log)).sum(dim=-1)  # [B, T]
    kl = kl * response_mask.float()
    n_tokens = response_mask.sum(dim=1).clamp(min=1)
    return (kl.sum(dim=1) / n_tokens).mean()


def _chunked_kl(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    response_mask: torch.Tensor,
    chunk_size: int,
    mode: str,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Compute KL-type divergence in chunks along sequence dimension."""
    B, T, V = p_logits.shape
    total_loss = torch.zeros(B, device=p_logits.device, dtype=torch.float32)
    n_tokens = response_mask.sum(dim=1).clamp(min=1)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        p_c = p_logits[:, start:end, :]
        q_c = q_logits[:, start:end, :]
        m_c = response_mask[:, start:end]

        p_log = F.log_softmax(p_c.float(), dim=-1)
        q_log = F.log_softmax(q_c.float(), dim=-1)
        p_prob = p_log.exp()
        q_prob = q_log.exp()

        if mode == "kl":
            kl = (p_prob * (p_log - q_log)).sum(dim=-1)
        elif mode == "skl":
            mix = alpha * p_prob + (1 - alpha) * q_prob
            kl = (p_prob * (p_log - mix.log())).sum(dim=-1)
        elif mode == "srkl":
            mix = (1 - alpha) * p_prob + alpha * q_prob
            kl = (q_prob * (q_log - mix.log())).sum(dim=-1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        total_loss += (kl * m_c.float()).sum(dim=1)

    return (total_loss / n_tokens).mean()
