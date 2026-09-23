"""Tests for layer mapping — verify Table 11 mappings for all 3 pairs."""

from __future__ import annotations

import pytest

from mta.models.layers import build_layer_pairs, build_strided_schedule


class TestGPT2Mapping:
    """GPT-2 120M (12 layers) → GPT-2 XL (48 layers)."""

    def test_layer_pairs(self):
        pairs = build_layer_pairs(
            word_layers=[6],
            phrase_layers=[9, 12],
            n_student=12,
            n_teacher=48,
        )
        assert len(pairs) == 3

        assert pairs[0].student_layer == 6
        assert pairs[0].teacher_layer == 24
        assert pairs[0].span_type == "word"

        assert pairs[1].student_layer == 9
        assert pairs[1].teacher_layer == 36
        assert pairs[1].span_type == "phrase"

        assert pairs[2].student_layer == 12
        assert pairs[2].teacher_layer == 48
        assert pairs[2].span_type == "phrase"


class TestQwenMapping:
    """Qwen1.5-0.5B (24 layers) → Qwen1.5-1.8B (24 layers)."""

    def test_identity_mapping(self):
        pairs = build_layer_pairs(
            word_layers=[14],
            phrase_layers=[16, 18, 20, 22, 24],
            n_student=24,
            n_teacher=24,
        )
        assert len(pairs) == 6
        for p in pairs:
            assert p.student_layer == p.teacher_layer, (
                f"Identity mapping expected: S{p.student_layer} != T{p.teacher_layer}"
            )


class TestOPTMapping:
    """OPT-1.3B (24 layers) → OPT-6.7B (32 layers)."""

    def test_layer_pairs(self):
        pairs = build_layer_pairs(
            word_layers=[16],
            phrase_layers=[18, 20, 22, 24],
            n_student=24,
            n_teacher=32,
        )
        expected = {
            (16, 21, "word"),
            (18, 24, "phrase"),
            (20, 26, "phrase"),
            (22, 29, "phrase"),
            (24, 32, "phrase"),
        }
        actual = {(p.student_layer, p.teacher_layer, p.span_type) for p in pairs}
        assert actual == expected, f"Mismatch:\n  expected: {expected}\n  actual:   {actual}"


class TestStridedSchedule:
    def test_basic(self):
        word, phrase = build_strided_schedule(n_student=12, stride=3, budget=3)
        assert word == [6]
        assert phrase == [9, 12]

    def test_budget_limits(self):
        word, phrase = build_strided_schedule(n_student=24, stride=2, budget=5)
        assert len(word) + len(phrase) == 5
        assert word[0] < phrase[0]
