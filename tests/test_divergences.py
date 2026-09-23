"""Tests for KL, SKL, SRKL divergences."""

from __future__ import annotations

import pytest
import torch

from mta.losses.divergences import kl_divergence, skew_kl, skew_reverse_kl


def _make_logits(B: int, T: int, V: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T, V)


def _make_response_mask(B: int, T: int, resp_start: int = 2) -> torch.Tensor:
    mask = torch.zeros(B, T)
    mask[:, resp_start:] = 1.0
    return mask


class TestKLZero:
    def test_identical_distributions(self):
        """KL = 0 when p == q."""
        logits = _make_logits(2, 8, 32)
        mask = _make_response_mask(2, 8)
        loss = kl_divergence(logits, logits, mask)
        assert loss.item() < 1e-6, f"KL should be ~0, got {loss.item()}"

    def test_skl_identical(self):
        """SKL = 0 when p == q."""
        logits = _make_logits(2, 8, 32)
        mask = _make_response_mask(2, 8)
        loss = skew_kl(logits, logits, mask, alpha=0.1)
        assert loss.item() < 1e-6, f"SKL should be ~0, got {loss.item()}"

    def test_srkl_identical(self):
        """SRKL = 0 when p == q."""
        logits = _make_logits(2, 8, 32)
        mask = _make_response_mask(2, 8)
        loss = skew_reverse_kl(logits, logits, mask, alpha=0.1)
        assert loss.item() < 1e-6, f"SRKL should be ~0, got {loss.item()}"


class TestChunkedEqualsNonChunked:
    def test_kl(self):
        logits_p = _make_logits(2, 16, 64, seed=1)
        logits_q = _make_logits(2, 16, 64, seed=2)
        mask = _make_response_mask(2, 16, 4)

        full = kl_divergence(logits_p, logits_q, mask, chunk_size=0)
        chunked = kl_divergence(logits_p, logits_q, mask, chunk_size=4)
        assert torch.allclose(full, chunked, atol=1e-5), f"Full={full.item()}, Chunked={chunked.item()}"

    def test_skl(self):
        logits_p = _make_logits(2, 16, 64, seed=1)
        logits_q = _make_logits(2, 16, 64, seed=2)
        mask = _make_response_mask(2, 16, 4)

        full = skew_kl(logits_p, logits_q, mask, alpha=0.1, chunk_size=0)
        chunked = skew_kl(logits_p, logits_q, mask, alpha=0.1, chunk_size=4)
        assert torch.allclose(full, chunked, atol=1e-5), f"Full={full.item()}, Chunked={chunked.item()}"

    def test_srkl(self):
        logits_p = _make_logits(2, 16, 64, seed=1)
        logits_q = _make_logits(2, 16, 64, seed=2)
        mask = _make_response_mask(2, 16, 4)

        full = skew_reverse_kl(logits_p, logits_q, mask, alpha=0.1, chunk_size=0)
        chunked = skew_reverse_kl(logits_p, logits_q, mask, alpha=0.1, chunk_size=4)
        assert torch.allclose(full, chunked, atol=1e-5), f"Full={full.item()}, Chunked={chunked.item()}"


class TestMaskResponse:
    def test_only_response_counts(self):
        """Loss should change only when response tokens differ."""
        B, T, V = 2, 10, 32
        logits_p = _make_logits(B, T, V, seed=1)
        logits_q = logits_p.clone()
        mask = _make_response_mask(B, T, resp_start=5)

        # Identical in response region
        loss_same = kl_divergence(logits_p, logits_q, mask)
        assert loss_same.item() < 1e-6

        # Differ only in prompt (should not affect loss)
        logits_q2 = logits_p.clone()
        logits_q2[:, :5, :] += 10.0
        loss_prompt_diff = kl_divergence(logits_p, logits_q2, mask)
        assert loss_prompt_diff.item() < 1e-6, "Prompt changes should not affect loss"

        # Differ in response (should affect loss) — scale, not shift,
        # because softmax is translation-invariant
        logits_q3 = logits_p.clone()
        logits_q3[:, 5:, :] *= 2.0
        loss_resp_diff = kl_divergence(logits_p, logits_q3, mask)
        assert loss_resp_diff.item() > 0.001, "Response changes should affect loss"


class TestNonNegativity:
    def test_kl_nonneg(self):
        logits_p = _make_logits(4, 16, 64, seed=1)
        logits_q = _make_logits(4, 16, 64, seed=2)
        mask = _make_response_mask(4, 16)
        loss = kl_divergence(logits_p, logits_q, mask)
        assert loss.item() >= -1e-6, f"KL should be non-negative, got {loss.item()}"

    def test_skl_nonneg(self):
        logits_p = _make_logits(4, 16, 64, seed=1)
        logits_q = _make_logits(4, 16, 64, seed=2)
        mask = _make_response_mask(4, 16)
        loss = skew_kl(logits_p, logits_q, mask, alpha=0.1)
        assert loss.item() >= -1e-6

    def test_srkl_nonneg(self):
        logits_p = _make_logits(4, 16, 64, seed=1)
        logits_q = _make_logits(4, 16, 64, seed=2)
        mask = _make_response_mask(4, 16)
        loss = skew_reverse_kl(logits_p, logits_q, mask, alpha=0.1)
        assert loss.item() >= -1e-6
