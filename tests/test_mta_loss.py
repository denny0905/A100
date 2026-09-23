"""Tests for MTA loss per spec requirements."""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn as nn

from mta.losses.mta import (
    MTALoss,
    compute_span_weights,
    compute_token_weights,
    span_pooling,
    _pad_spans_batch,
)
from mta.models.layers import LayerPair


def _make_hidden(B: int, T: int, D: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T, D)


def _make_mask(B: int, T: int, valid_lens: list[int]) -> torch.Tensor:
    mask = torch.zeros(B, T, dtype=torch.float32)
    for i, vl in enumerate(valid_lens):
        mask[i, :vl] = 1.0
    return mask


def _make_spans(word_spans: list[tuple[int, int]], phrase_spans: list[tuple[int, int]]) -> dict[str, torch.LongTensor]:
    w = torch.tensor(word_spans, dtype=torch.long) if word_spans else torch.zeros(0, 2, dtype=torch.long)
    p = torch.tensor(phrase_spans, dtype=torch.long) if phrase_spans else torch.zeros(0, 2, dtype=torch.long)
    return {"word": w, "phrase": p}


class TestTokenWeights:
    def test_sum_approx_one(self):
        """Token weights should sum to ~1 over valid tokens."""
        B, T, D = 2, 16, 32
        h = _make_hidden(B, T, D)
        mask = _make_mask(B, T, [16, 10])
        w = compute_token_weights(h, mask)

        for i in range(B):
            valid = mask[i].sum().int().item()
            total = w[i, :valid].sum().item()
            assert abs(total - 1.0) < 0.01, f"Sample {i}: sum={total}, expected ~1.0"

    def test_padding_zero(self):
        """Padding token weights should be exactly 0."""
        B, T, D = 2, 16, 32
        h = _make_hidden(B, T, D)
        mask = _make_mask(B, T, [12, 8])
        w = compute_token_weights(h, mask)

        for i in range(B):
            valid = int(mask[i].sum().item())
            assert (w[i, valid:] == 0).all(), f"Sample {i}: non-zero weight in padding"

    def test_invariant_to_padding(self):
        """Weights on valid tokens should not change when more padding is added."""
        B, D = 1, 32
        h_short = _make_hidden(B, 10, D, seed=42)
        mask_short = _make_mask(B, 10, [10])
        w_short = compute_token_weights(h_short, mask_short)

        h_long = torch.cat([h_short, torch.randn(B, 6, D)], dim=1)
        mask_long = _make_mask(B, 16, [10])
        w_long = compute_token_weights(h_long, mask_long)

        assert torch.allclose(w_short[0, :10], w_long[0, :10], atol=1e-5), \
            "Weights changed when padding was added"

    def test_chunked_equals_full(self):
        """Chunked computation should match full computation."""
        B, T, D = 2, 20, 32
        h = _make_hidden(B, T, D)
        mask = _make_mask(B, T, [20, 15])

        w_full = compute_token_weights(h, mask, chunk_size=0)
        w_chunk = compute_token_weights(h, mask, chunk_size=8)

        assert torch.allclose(w_full, w_chunk, atol=1e-5), "Chunked != full"


class TestSpanPooling:
    def test_manual_weighted_average(self):
        """Pooling should match hand-computed weighted average."""
        B, T, D = 1, 6, 4
        h = torch.tensor([[[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [1, 1, 0, 0],
                           [0, 0, 1, 1]]], dtype=torch.float32)  # [1, 6, 4]

        # Span covers tokens 1-3 (indices 1, 2)
        spans = torch.tensor([[[1, 3]]], dtype=torch.long)  # [1, 1, 2]
        mask = torch.ones(1, 1, dtype=torch.bool)

        w = torch.tensor([[0.1, 0.3, 0.2, 0.1, 0.2, 0.1]])  # [1, 6]

        pooled = span_pooling(h, w, spans, mask)  # [1, 1, 4]

        # Manual: within span [1, 3): tokens 1 (w=0.3) and 2 (w=0.2)
        # pool_w_1 = 0.3 / (0.3+0.2) = 0.6, pool_w_2 = 0.2 / 0.5 = 0.4
        expected = 0.6 * h[0, 1] + 0.4 * h[0, 2]
        assert torch.allclose(pooled[0, 0], expected, atol=1e-5), \
            f"Expected {expected}, got {pooled[0, 0]}"


class TestDSA:
    def test_zero_when_identical(self):
        """L_DSA = 0 when student hidden == teacher hidden (same dim)."""
        B, T, D = 2, 16, 32
        h = _make_hidden(B, T, D)
        mask = _make_mask(B, T, [16, 16])
        spans = [
            _make_spans([(0, 3), (4, 7), (8, 11)], [(0, 5), (6, 11)]),
            _make_spans([(0, 3), (4, 7), (8, 11)], [(0, 5), (6, 11)]),
        ]

        # Use identity projector (same dim)
        pairs = [LayerPair(1, 1, "word"), LayerPair(2, 2, "phrase")]
        loss_mod = MTALoss(d_student=D, d_teacher=D, layer_pairs=pairs)

        # Set projectors to identity
        for key in loss_mod.projectors:
            loss_mod.projectors[key].weight.data = torch.eye(D)

        # hidden_states: tuple indexed by layer
        hidden = tuple(h.clone() for _ in range(3))

        result = loss_mod(hidden, hidden, mask, spans)
        assert result["dsa"].item() < 1e-6, f"DSA should be ~0, got {result['dsa'].item()}"

    def test_positive_with_noise(self):
        """L_DSA > 0 when student has noise added."""
        B, T, D = 2, 16, 32
        h_t = _make_hidden(B, T, D, seed=1)
        h_s = h_t + torch.randn_like(h_t) * 0.5
        mask = _make_mask(B, T, [16, 16])
        spans = [
            _make_spans([(0, 3), (4, 7), (8, 11)], [(0, 5), (6, 11)]),
            _make_spans([(0, 3), (4, 7), (8, 11)], [(0, 5), (6, 11)]),
        ]

        pairs = [LayerPair(1, 1, "word"), LayerPair(2, 2, "phrase")]
        loss_mod = MTALoss(d_student=D, d_teacher=D, layer_pairs=pairs)
        for key in loss_mod.projectors:
            loss_mod.projectors[key].weight.data = torch.eye(D)

        t_hidden = tuple(h_t.clone() for _ in range(3))
        s_hidden = tuple(h_s.clone() for _ in range(3))

        result = loss_mod(s_hidden, t_hidden, mask, spans)
        assert result["dsa"].item() > 0, "DSA should be > 0 with noise"


class TestHid:
    def test_zero_when_identical(self):
        """L_Hid ≈ 0 when projector is identity and hidden match."""
        B, T, D = 2, 16, 32
        h = _make_hidden(B, T, D)
        mask = _make_mask(B, T, [16, 16])
        spans = [
            _make_spans([(0, 4), (5, 9)], [(0, 8)]),
            _make_spans([(0, 4), (5, 9)], [(0, 8)]),
        ]

        pairs = [LayerPair(1, 1, "word"), LayerPair(2, 2, "phrase")]
        loss_mod = MTALoss(d_student=D, d_teacher=D, layer_pairs=pairs)
        for key in loss_mod.projectors:
            loss_mod.projectors[key].weight.data = torch.eye(D)

        hidden = tuple(h.clone() for _ in range(3))
        result = loss_mod(hidden, hidden, mask, spans)
        assert result["hid"].item() < 1e-5, f"Hid should be ~0, got {result['hid'].item()}"


class TestGradientFlow:
    def test_gradient_only_student_and_projector(self):
        """Gradient should flow to student and projector, not teacher."""
        B, T, D_s, D_t = 2, 16, 16, 32
        h_s = _make_hidden(B, T, D_s).requires_grad_(True)
        h_t = _make_hidden(B, T, D_t)
        mask = _make_mask(B, T, [16, 16])
        spans = [
            _make_spans([(0, 4), (5, 9)], [(0, 8), (9, 14)]),
            _make_spans([(0, 4), (5, 9)], [(0, 8), (9, 14)]),
        ]

        pairs = [LayerPair(1, 1, "word"), LayerPair(2, 2, "phrase")]
        loss_mod = MTALoss(d_student=D_s, d_teacher=D_t, layer_pairs=pairs)

        s_hidden = (torch.zeros(B, T, D_s), h_s, h_s.clone().requires_grad_(True))
        t_hidden = tuple(h_t.clone() for _ in range(3))

        result = loss_mod(s_hidden, t_hidden, mask, spans)
        total = result["dsa"] + result["hid"]
        total.backward()

        assert h_s.grad is not None, "Student should have gradients"
        assert h_s.grad.abs().sum() > 0, "Student gradients should be non-zero"

        for name, param in loss_mod.projectors.named_parameters():
            assert param.grad is not None, f"Projector {name} should have gradients"

        # Teacher hidden should not have grad (not requires_grad)
        assert not h_t.requires_grad


class TestEdgeCases:
    def test_zero_spans(self):
        """No NaN when a sample has 0 spans."""
        B, T, D = 2, 16, 32
        h = _make_hidden(B, T, D)
        mask = _make_mask(B, T, [16, 16])
        spans = [
            _make_spans([], []),  # no spans
            _make_spans([(0, 4)], [(0, 8)]),
        ]

        pairs = [LayerPair(1, 1, "word"), LayerPair(2, 2, "phrase")]
        loss_mod = MTALoss(d_student=D, d_teacher=D, layer_pairs=pairs)

        hidden = tuple(h.clone() for _ in range(3))
        result = loss_mod(hidden, hidden, mask, spans)
        assert not torch.isnan(result["dsa"]), "DSA should not be NaN with 0 spans"
        assert not torch.isnan(result["hid"]), "Hid should not be NaN with 0 spans"

    def test_one_span(self):
        """No NaN when a sample has exactly 1 span (DSA needs ≥2)."""
        B, T, D = 1, 16, 32
        h = _make_hidden(B, T, D)
        mask = _make_mask(B, T, [16])
        spans = [_make_spans([(0, 4)], [(0, 8)])]

        pairs = [LayerPair(1, 1, "word")]
        loss_mod = MTALoss(d_student=D, d_teacher=D, layer_pairs=pairs)

        hidden = tuple(h.clone() for _ in range(2))
        result = loss_mod(hidden, hidden, mask, spans)
        assert not torch.isnan(result["dsa"])


class TestPerformance:
    def test_vectorized_reasonable_time(self):
        """Batch 16 × 512 tokens × 256 spans should complete in reasonable time on CPU."""
        B, T, D = 16, 512, 64
        K_word, K_phrase = 128, 64
        h = _make_hidden(B, T, D)
        mask = _make_mask(B, T, [T] * B)

        spans = []
        for _ in range(B):
            word_s = [(i * 3, min(i * 3 + 3, T)) for i in range(K_word)]
            phrase_s = [(i * 7, min(i * 7 + 7, T)) for i in range(K_phrase)]
            spans.append(_make_spans(word_s, phrase_s))

        pairs = [LayerPair(1, 1, "word"), LayerPair(2, 2, "phrase")]
        loss_mod = MTALoss(d_student=D, d_teacher=D, layer_pairs=pairs)
        for key in loss_mod.projectors:
            loss_mod.projectors[key].weight.data = torch.eye(D)

        hidden = tuple(h.clone() for _ in range(3))

        start = time.time()
        result = loss_mod(hidden, hidden, mask, spans)
        elapsed = time.time() - start

        assert elapsed < 60, f"Took {elapsed:.1f}s — should be under 60s on CPU"
        assert not torch.isnan(result["dsa"])
        assert not torch.isnan(result["hid"])
