"""Generate summary tables comparing results to paper numbers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)

PAPER_NUMBERS = {
    "gpt2": {
        "Teacher": {"dolly_test": 28.71, "selfinst_test": 15.68, "vicuna_test": 17.00, "sni_test": 28.80},
        "sft": {"dolly_test": 23.33, "selfinst_test": 10.56, "vicuna_test": 15.12, "sni_test": 17.08},
        "fdd": {"dolly_test": 25.47, "selfinst_test": 12.45, "vicuna_test": 16.44, "sni_test": 23.54},
        "fdd+mta": {"dolly_test": 25.64, "selfinst_test": 13.60, "vicuna_test": 17.00, "sni_test": 25.75},
        "distillm": {"dolly_test": 25.65, "selfinst_test": 13.39, "vicuna_test": 16.50, "sni_test": 25.28},
        "distillm+mta": {"dolly_test": 25.77, "selfinst_test": 14.19, "vicuna_test": 16.67, "sni_test": 29.18},
        "distillm2": {"dolly_test": 22.44, "selfinst_test": 12.52, "vicuna_test": 12.30, "sni_test": 27.10},
        "distillm2+mta": {"dolly_test": 24.76, "selfinst_test": 14.16, "vicuna_test": 14.28, "sni_test": 26.54},
    },
    "qwen": {
        "Teacher": {"dolly_test": 28.23, "selfinst_test": 19.58, "vicuna_test": 19.59, "sni_test": 34.36},
        "fdd": {"dolly_test": 25.08, "selfinst_test": 12.24, "vicuna_test": 16.08, "sni_test": 23.69},
        "fdd+mta": {"dolly_test": 25.35, "selfinst_test": 13.85, "vicuna_test": 17.21, "sni_test": 27.25},
        "distillm": {"dolly_test": 25.16, "selfinst_test": 12.90, "vicuna_test": 15.86, "sni_test": 25.28},
        "distillm+mta": {"dolly_test": 25.61, "selfinst_test": 13.08, "vicuna_test": 16.04, "sni_test": 29.32},
        "distillm2": {"dolly_test": 27.48, "selfinst_test": 17.95, "vicuna_test": 17.14, "sni_test": 30.99},
        "distillm2+mta": {"dolly_test": 27.93, "selfinst_test": 18.87, "vicuna_test": 18.63, "sni_test": 33.49},
    },
    "opt": {
        "Teacher": {"dolly_test": 27.60, "selfinst_test": 16.40, "vicuna_test": 17.80, "sni_test": 30.30},
        "fdd": {"dolly_test": 26.07, "selfinst_test": 14.82, "vicuna_test": 17.09, "sni_test": 28.98},
        "fdd+mta": {"dolly_test": 26.49, "selfinst_test": 16.47, "vicuna_test": 17.51, "sni_test": 31.12},
        "distillm": {"dolly_test": 27.87, "selfinst_test": 15.99, "vicuna_test": 18.02, "sni_test": 30.02},
        "distillm+mta": {"dolly_test": 27.43, "selfinst_test": 17.07, "vicuna_test": 18.40, "sni_test": 32.97},
        "distillm2": {"dolly_test": 26.31, "selfinst_test": 17.62, "vicuna_test": 17.60, "sni_test": 30.85},
        "distillm2+mta": {"dolly_test": 26.79, "selfinst_test": 18.16, "vicuna_test": 16.55, "sni_test": 31.39},
    },
}

TEST_SET_NAMES = {
    "dolly_test": "Dolly",
    "selfinst_test": "SelfInst",
    "vicuna_test": "Vicuna",
    "sni_test": "S-NI",
}


def _collect_scores(results_dir: Path, experiment: str, test_sets: list[str], seeds: list[int]) -> dict[str, dict]:
    """Collect ROUGE-L scores across seeds for an experiment."""
    exp_dir = results_dir / experiment
    scores: dict[str, list[float]] = {ts: [] for ts in test_sets}

    for ts in test_sets:
        for seed in seeds:
            f = exp_dir / f"{ts}_seed{seed}.jsonl"
            if not f.exists():
                continue
            with f.open("r", encoding="utf-8") as fp:
                items = [json.loads(line) for line in fp if line.strip()]
            if items:
                avg = sum(it["rouge_l"] for it in items) / len(items)
                scores[ts].append(avg)

    result = {}
    for ts in test_sets:
        vals = scores[ts]
        if vals:
            result[ts] = {"mean": np.mean(vals), "std": np.std(vals)}
        else:
            result[ts] = {"mean": float("nan"), "std": float("nan")}
    return result


def generate_report(cfg: DictConfig) -> None:
    """Generate summary.md and summary.csv."""
    pair = cfg.pair_name
    results_dir = Path("results") / pair
    out_dir = results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    test_sets = list(cfg.eval.test_sets)
    seeds = list(cfg.eval.seeds)
    experiments = list(cfg.experiments)

    paper = PAPER_NUMBERS.get(pair, {})
    ts_headers = [TEST_SET_NAMES.get(ts, ts) for ts in test_sets]

    lines = [f"# Results: {pair}\n"]
    lines.append(f"| Method | {' | '.join(ts_headers)} | Avg |")
    lines.append("|" + "---|" * (len(ts_headers) + 2))

    # Paper reference rows
    if paper:
        for method, paper_scores in paper.items():
            vals = [paper_scores.get(ts, float("nan")) for ts in test_sets]
            avg = np.nanmean(vals)
            row = f"| {method} (paper) | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {avg:.2f} |"
            lines.append(row)
        lines.append("|---|" * (len(ts_headers) + 2))

    # Our results
    csv_rows = [",".join(["Method"] + [TEST_SET_NAMES.get(ts, ts) for ts in test_sets] + ["Avg"])]

    for exp in experiments:
        scores = _collect_scores(results_dir, exp, test_sets, seeds)
        means = [scores[ts]["mean"] for ts in test_sets]
        stds = [scores[ts]["std"] for ts in test_sets]
        avg_mean = np.nanmean(means)

        cells = []
        for m, s in zip(means, stds):
            if np.isnan(m):
                cells.append("—")
            else:
                cells.append(f"{m:.2f}±{s:.2f}")

        # Delta from paper
        paper_exp = paper.get(exp, {})
        deltas = []
        for ts, m in zip(test_sets, means):
            p = paper_exp.get(ts, float("nan"))
            if not np.isnan(m) and not np.isnan(p):
                deltas.append(m - p)
        avg_delta = np.mean(deltas) if deltas else float("nan")
        delta_str = f" (Δ{avg_delta:+.2f})" if not np.isnan(avg_delta) else ""

        row = f"| {exp} | " + " | ".join(cells) + f" | {avg_mean:.2f}{delta_str} |"
        lines.append(row)

        csv_vals = [f"{m:.2f}" if not np.isnan(m) else "" for m in means]
        csv_rows.append(",".join([exp] + csv_vals + [f"{avg_mean:.2f}"]))

    md_content = "\n".join(lines) + "\n"
    csv_content = "\n".join(csv_rows) + "\n"

    (out_dir / "summary.md").write_text(md_content, encoding="utf-8")
    (out_dir / "summary.csv").write_text(csv_content, encoding="utf-8")
    log.info("Report written to %s/summary.md", out_dir)
