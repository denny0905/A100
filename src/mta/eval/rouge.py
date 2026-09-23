"""ROUGE-L computation following MiniLLM protocol."""

from __future__ import annotations

from rouge_score import rouge_scorer


_scorer = None


def _get_scorer() -> rouge_scorer.RougeScorer:
    global _scorer
    if _scorer is None:
        _scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _scorer


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score (×100 to match paper scale)."""
    scorer = _get_scorer()
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure * 100.0
