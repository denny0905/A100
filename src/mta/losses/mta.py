"""MTA loss: token weight, span weight, pooling, L_DSA, L_Hid.

All computation in fp32. Fully vectorized — no Python loops over spans or tokens.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from mta.models.layers import LayerPair

log = logging.getLogger(__name__)


def compute_token_weights(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_size: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute token importance weights from hidden states (Eq. 5-7, Appendix C).

    Args:
        hidden: [B, T, D] teacher hidden states at a specific layer (fp32).
        attention_mask: [B, T] with 1 for valid, 0 for padding.
        chunk_size: if >0, compute attention in chunks of this size along the row dim.
        eps: numerical stability.

    Returns:
        weights: [B, T] token importance weights. Sum ≈ 1 over valid tokens per sample.
    """
    B, T, D = hidden.shape

    # Eq. 5: standardize by std only (no mean subtraction)
    std = hidden.std(dim=-1, keepdim=True) + eps
    h_norm = hidden / std  # [B, T, D]

    # Build mask: -inf for diagonal and padding columns (Eq. 21)
    # pad_mask[b, t] = 1 where t is padding
    pad_mask = (1 - attention_mask).bool()  # [B, T]

    if chunk_size <= 0 or chunk_size >= T:
        # Full attention matrix
        # Eq. 6: S = (H_norm @ H_norm^T) / sqrt(d)
        scores = torch.bmm(h_norm, h_norm.transpose(1, 2)) / (D ** 0.5)  # [B, T, T]

        # Diagonal mask: -inf on diagonal
        diag_mask = torch.eye(T, device=hidden.device, dtype=torch.bool).unsqueeze(0)  # [1, T, T]
        scores = scores.masked_fill(diag_mask, float("-inf"))

        # Padding column mask: -inf for all columns that are padding
        col_pad = pad_mask.unsqueeze(1).expand_as(scores)  # [B, T, T]
        scores = scores.masked_fill(col_pad, float("-inf"))

        # Eq. 22: softmax along columns (dim=-1)
        alpha = torch.softmax(scores, dim=-1)  # [B, T, T]

        # Zero out padding rows (D03)
        row_pad = pad_mask.unsqueeze(2).expand_as(alpha)  # [B, T, T]
        alpha = alpha.masked_fill(row_pad, 0.0)

        # Eq. 23: w_t = (1/N_valid) * sum_s alpha[s,t]
        n_valid = attention_mask.sum(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1)  # [B, 1, 1]
        weights = alpha.sum(dim=1) / n_valid.squeeze(-1)  # [B, T]
    else:
        # Chunked computation to save memory
        weights = torch.zeros(B, T, device=hidden.device, dtype=hidden.dtype)
        n_valid = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, 1]

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            # Compute chunk of rows [start:end] attending to all T columns
            h_chunk = h_norm[:, start:end, :]  # [B, chunk, D]
            scores = torch.bmm(h_chunk, h_norm.transpose(1, 2)) / (D ** 0.5)  # [B, chunk, T]

            # Diagonal mask for this chunk
            chunk_len = end - start
            diag = torch.zeros(chunk_len, T, device=hidden.device, dtype=torch.bool)
            for i in range(chunk_len):
                diag[i, start + i] = True
            scores = scores.masked_fill(diag.unsqueeze(0), float("-inf"))

            # Padding column mask
            col_pad = pad_mask.unsqueeze(1).expand(B, chunk_len, T)
            scores = scores.masked_fill(col_pad, float("-inf"))

            alpha_chunk = torch.softmax(scores, dim=-1)  # [B, chunk, T]

            # Zero out padding rows in chunk
            row_pad = pad_mask[:, start:end].unsqueeze(2).expand_as(alpha_chunk)
            alpha_chunk = alpha_chunk.masked_fill(row_pad, 0.0)

            # Accumulate: w_t += sum over rows in this chunk
            weights += alpha_chunk.sum(dim=1)  # [B, T]

        weights = weights / n_valid  # [B, T]

    # Ensure padding tokens have exactly 0 weight
    weights = weights * attention_mask.float()
    return weights


def compute_span_weights(
    token_weights: torch.Tensor,
    spans: torch.LongTensor,
    span_mask: torch.BoolTensor,
) -> torch.Tensor:
    """Compute span importance weights (Eq. 9).

    Args:
        token_weights: [B, T]
        spans: [B, K, 2] start/end indices
        span_mask: [B, K] True where span is valid

    Returns:
        span_weights: [B, K] normalized span weights (sum to 1 per sample where valid)
    """
    B, K, _ = spans.shape
    T = token_weights.shape[1]

    # Build token-to-span membership: [B, K, T]
    positions = torch.arange(T, device=spans.device).view(1, 1, T)  # [1, 1, T]
    starts = spans[:, :, 0:1]  # [B, K, 1]
    ends = spans[:, :, 1:2]    # [B, K, 1]
    membership = (positions >= starts) & (positions < ends)  # [B, K, T]
    membership = membership & span_mask.unsqueeze(2)  # mask invalid spans

    # w_tilde_k = sum_{t in span_k} w_t
    w_expanded = token_weights.unsqueeze(1).expand(B, K, T)  # [B, K, T]
    w_tilde = (w_expanded * membership.float()).sum(dim=2)  # [B, K]

    # Normalize within each sample
    w_sum = w_tilde.sum(dim=1, keepdim=True).clamp(min=1e-8)
    span_w = w_tilde / w_sum  # [B, K]
    span_w = span_w * span_mask.float()

    return span_w


def span_pooling(
    hidden: torch.Tensor,
    token_weights: torch.Tensor,
    spans: torch.LongTensor,
    span_mask: torch.BoolTensor,
) -> torch.Tensor:
    """Weighted average pooling of hidden states within spans (Eq. 8).

    Args:
        hidden: [B, T, D]
        token_weights: [B, T]
        spans: [B, K, 2]
        span_mask: [B, K]

    Returns:
        pooled: [B, K, D] span representations
    """
    B, T, D = hidden.shape
    K = spans.shape[1]

    # Build membership and within-span weights
    positions = torch.arange(T, device=spans.device).view(1, 1, T)
    starts = spans[:, :, 0:1]
    ends = spans[:, :, 1:2]
    membership = (positions >= starts) & (positions < ends) & span_mask.unsqueeze(2)

    w_expanded = token_weights.unsqueeze(1).expand(B, K, T)  # [B, K, T]
    w_masked = w_expanded * membership.float()  # [B, K, T]

    # Normalize within each span
    w_sum = w_masked.sum(dim=2, keepdim=True).clamp(min=1e-8)  # [B, K, 1]
    pool_weights = w_masked / w_sum  # [B, K, T] — sums to 1 within each span

    # Weighted average: U_k = sum_t pool_w[k,t] * H[t]
    pooled = torch.bmm(pool_weights, hidden)  # [B, K, D]

    # Zero out invalid spans
    pooled = pooled * span_mask.unsqueeze(2).float()
    return pooled


def _pad_spans_batch(
    span_list: list[torch.LongTensor],
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    """Pad variable-length span lists into a batch.

    Args:
        span_list: list of [K_i, 2] tensors

    Returns:
        spans: [B, K_max, 2]
        mask: [B, K_max]
    """
    max_k = max(s.shape[0] for s in span_list) if span_list else 0
    device = span_list[0].device if span_list else torch.device("cpu")
    if max_k == 0:
        B = len(span_list)
        return (
            torch.zeros(B, 0, 2, dtype=torch.long, device=device),
            torch.zeros(B, 0, dtype=torch.bool, device=device),
        )

    padded = []
    masks = []
    for s in span_list:
        k = s.shape[0]
        pad_len = max_k - k
        if pad_len > 0:
            padded.append(torch.cat([s, torch.zeros(pad_len, 2, dtype=torch.long, device=s.device)], dim=0))
        else:
            padded.append(s)
        mask = torch.cat([
            torch.ones(k, dtype=torch.bool, device=s.device),
            torch.zeros(pad_len, dtype=torch.bool, device=s.device),
        ])
        masks.append(mask)

    return torch.stack(padded), torch.stack(masks)


class MTALoss(nn.Module):
    """Multi-Granular Trajectory Alignment loss (Eq. 10-14)."""

    def __init__(
        self,
        d_student: int,
        d_teacher: int,
        layer_pairs: list[LayerPair],
        student_pool_weights: str = "teacher",
        hid_renorm: bool = False,
        chunk_size: int = 0,
    ) -> None:
        super().__init__()
        self.layer_pairs = layer_pairs
        self.student_pool_weights = student_pool_weights
        self.hid_renorm = hid_renorm
        self.chunk_size = chunk_size

        # One projector per student layer (shared if same layer appears in word+phrase)
        self.projectors = nn.ModuleDict()
        for lp in layer_pairs:
            key = str(lp.student_layer)
            if key not in self.projectors:
                self.projectors[key] = nn.Linear(d_student, d_teacher, bias=False)

    def forward(
        self,
        s_hidden: tuple[torch.Tensor, ...],
        t_hidden: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        spans_batch: list[dict[str, torch.LongTensor]],
    ) -> dict[str, torch.Tensor]:
        """Compute MTA losses.

        Args:
            s_hidden: student hidden_states tuple (index 0 = embedding).
            t_hidden: teacher hidden_states tuple.
            attention_mask: [B, T].
            spans_batch: list of dicts with 'word' and 'phrase' keys, each [K_i, 2].

        Returns:
            dict with 'dsa', 'hid', and per-layer values for logging.
        """
        device = attention_mask.device
        B = attention_mask.shape[0]

        # Collect spans per type
        word_spans_list = [s["word"].to(device) for s in spans_batch]
        phrase_spans_list = [s["phrase"].to(device) for s in spans_batch]

        word_spans, word_mask = _pad_spans_batch(word_spans_list)
        phrase_spans, phrase_mask = _pad_spans_batch(phrase_spans_list)
        word_spans = word_spans.to(device)
        word_mask = word_mask.to(device)
        phrase_spans = phrase_spans.to(device)
        phrase_mask = phrase_mask.to(device)

        results: dict[str, torch.Tensor] = {}
        dsa_losses = []
        hid_losses = []

        for lp in self.layer_pairs:
            h_s = s_hidden[lp.student_layer].float()  # [B, T, d_s]
            h_t = t_hidden[lp.teacher_layer].float()  # [B, T, d_t]

            # Select spans for this layer's type
            if lp.span_type == "word":
                spans_padded, s_mask = word_spans, word_mask
            else:
                spans_padded, s_mask = phrase_spans, phrase_mask

            # Token weights from teacher (within no_grad)
            with torch.no_grad():
                t_weights = compute_token_weights(h_t, attention_mask.float(), self.chunk_size)

            # Pool teacher spans
            with torch.no_grad():
                t_span_w = compute_span_weights(t_weights, spans_padded, s_mask)
                t_pooled = span_pooling(h_t, t_weights, spans_padded, s_mask)

            # Pool student spans
            if self.student_pool_weights == "teacher":
                pool_w_for_student = t_weights
            else:
                with torch.no_grad():
                    pool_w_for_student = compute_token_weights(
                        h_s.detach(), attention_mask.float(), self.chunk_size
                    )

            s_pooled = span_pooling(h_s, pool_w_for_student, spans_padded, s_mask)

            # --- L_DSA (Eq. 10-11) ---
            dsa_l = self._compute_dsa(s_pooled, t_pooled, t_span_w, s_mask)
            dsa_losses.append(dsa_l)
            results[f"dsa_l{lp.student_layer}"] = dsa_l.detach()

            # --- L_Hid (Eq. 12-13) ---
            proj = self.projectors[str(lp.student_layer)]
            h_s_proj = proj(h_s)  # [B, T, d_t]

            hid_l = self._compute_hid(
                h_s_proj, h_t, t_weights, spans_padded, s_mask, attention_mask,
            )
            hid_losses.append(hid_l)
            results[f"hid_l{lp.student_layer}"] = hid_l.detach()

        # Aggregate: DSA averaged over layers (Eq. 10), Hid summed (Eq. 13)
        if dsa_losses:
            results["dsa"] = torch.stack(dsa_losses).mean()
        else:
            results["dsa"] = torch.tensor(0.0, device=device)

        if hid_losses:
            results["hid"] = torch.stack(hid_losses).sum()
        else:
            results["hid"] = torch.tensor(0.0, device=device)

        return results

    @staticmethod
    def _compute_dsa(
        s_pooled: torch.Tensor,
        t_pooled: torch.Tensor,
        span_weights: torch.Tensor,
        span_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """L_DSA for one layer (Eq. 11)."""
        B, K, D = s_pooled.shape

        if K < 2:
            return torch.tensor(0.0, device=s_pooled.device, dtype=s_pooled.dtype)

        # Normalize for cosine distance
        s_norm = F.normalize(s_pooled, dim=-1)  # [B, K, D]
        t_norm = F.normalize(t_pooled, dim=-1)

        # Pairwise cosine distance matrices
        d_s = 1.0 - torch.bmm(s_norm, s_norm.transpose(1, 2))  # [B, K, K]
        d_t = 1.0 - torch.bmm(t_norm, t_norm.transpose(1, 2))  # [B, K, K]

        # Pairwise span weight: w_ij = w_i * w_j
        w_ij = span_weights.unsqueeze(2) * span_weights.unsqueeze(1)  # [B, K, K]

        # Upper triangular mask (i < j) combined with valid span pairs
        triu = torch.triu(torch.ones(K, K, device=s_pooled.device, dtype=torch.bool), diagonal=1)
        pair_mask = triu.unsqueeze(0) & span_mask.unsqueeze(2) & span_mask.unsqueeze(1)

        # Loss: sum over valid pairs
        diff_sq = (d_s - d_t) ** 2  # [B, K, K]
        loss = (w_ij * diff_sq * pair_mask.float()).sum(dim=(1, 2))  # [B]

        return loss.mean()

    def _compute_hid(
        self,
        h_s_proj: torch.Tensor,
        h_t: torch.Tensor,
        t_weights: torch.Tensor,
        spans: torch.LongTensor,
        span_mask: torch.BoolTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """L_Hid for one layer (Eq. 13)."""
        B, T, D = h_s_proj.shape

        # Build token coverage mask M_l: tokens covered by any span at this layer
        positions = torch.arange(T, device=spans.device).view(1, 1, T)  # [1, 1, T]
        starts = spans[:, :, 0:1]  # [B, K, 1]
        ends = spans[:, :, 1:2]    # [B, K, 1]
        in_span = (positions >= starts) & (positions < ends) & span_mask.unsqueeze(2)
        token_covered = in_span.any(dim=1)  # [B, T]
        token_covered = token_covered & attention_mask.bool()

        if not token_covered.any():
            return torch.tensor(0.0, device=h_s_proj.device, dtype=h_s_proj.dtype)

        # Cosine distance per token
        cos_dist = 1.0 - F.cosine_similarity(h_s_proj, h_t, dim=-1)  # [B, T]

        # Weight by teacher token weights
        weights = t_weights.clone()
        if self.hid_renorm:
            # Renormalize weights over covered tokens
            w_covered = weights * token_covered.float()
            w_sum = w_covered.sum(dim=1, keepdim=True).clamp(min=1e-8)
            weights = w_covered / w_sum
        else:
            weights = weights * token_covered.float()

        loss = (weights * cos_dist).sum(dim=1)  # [B]
        return loss.mean()
