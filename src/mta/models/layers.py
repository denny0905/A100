"""Layer mapping between teacher and student for MTA alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass
class LayerPair:
    student_layer: int
    teacher_layer: int
    span_type: Literal["word", "phrase"]


def compute_teacher_layer(student_layer: int, n_student: int, n_teacher: int) -> int:
    """Map student layer to teacher layer: l_T = floor(l_S * N_T / N_S)."""
    return math.floor(student_layer * n_teacher / n_student)


def build_layer_pairs(
    word_layers: list[int],
    phrase_layers: list[int],
    n_student: int,
    n_teacher: int,
) -> list[LayerPair]:
    """Build (student, teacher, span_type) triples from config."""
    pairs = []
    for l_s in word_layers:
        l_t = compute_teacher_layer(l_s, n_student, n_teacher)
        pairs.append(LayerPair(l_s, l_t, "word"))
    for l_s in phrase_layers:
        l_t = compute_teacher_layer(l_s, n_student, n_teacher)
        pairs.append(LayerPair(l_s, l_t, "phrase"))
    return pairs


def build_strided_schedule(
    n_student: int,
    stride: int,
    budget: int,
) -> tuple[list[int], list[int]]:
    """Generate top-down strided layer schedule for ablation (Table 5).

    Returns (word_layers, phrase_layers): lowest layer is word, rest are phrase.
    """
    layers = []
    l = n_student
    while len(layers) < budget and l > 0:
        layers.append(l)
        l -= stride
    layers.sort()
    return layers[:1], layers[1:]
