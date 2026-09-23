"""Optional GPT-4o-mini judge (Appendix F)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from omegaconf import DictConfig

log = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an expert evaluator. Compare the following response to the reference answer.

Instruction: {instruction}

Reference Answer: {reference}

Model Response: {response}

Rate the model response on a scale of 1-5:
1 = Completely wrong or irrelevant
2 = Partially addresses the instruction but mostly incorrect
3 = Addresses the instruction but with significant issues
4 = Good response with minor issues
5 = Excellent, fully addresses the instruction

Output your rating in the format: [[rating]]"""


def run_judge(cfg: DictConfig) -> None:
    """Run GPT-4o-mini judge on generated outputs."""
    if not cfg.judge.enabled:
        log.info("Judge disabled, skipping.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.warning("OPENAI_API_KEY not set, skipping judge.")
        return

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
    except ImportError:
        log.warning("openai package not installed, skipping judge.")
        return

    results_dir = Path("results") / cfg.pair_name
    for exp in cfg.experiments:
        exp_dir = results_dir / exp
        if not exp_dir.exists():
            continue

        for test_file in exp_dir.glob("*_seed*.jsonl"):
            judge_file = test_file.with_suffix(".judge.jsonl")
            if judge_file.exists():
                continue

            with test_file.open("r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f if line.strip()]

            judge_results = []
            for item in items:
                prompt = JUDGE_PROMPT.format(
                    instruction=item.get("prompt", ""),
                    reference=item.get("reference", ""),
                    response=item.get("prediction", ""),
                )

                try:
                    resp = client.chat.completions.create(
                        model=cfg.judge.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=50,
                    )
                    text = resp.choices[0].message.content or ""
                    # Parse [[rating]]
                    import re
                    match = re.search(r"\[\[(\d)\]\]", text)
                    rating = int(match.group(1)) if match else 0
                except Exception as e:
                    log.warning("Judge API error: %s", e)
                    rating = 0

                judge_results.append({**item, "judge_rating": rating})

            with judge_file.open("w", encoding="utf-8") as f:
                for r in judge_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            avg_rating = sum(r["judge_rating"] for r in judge_results) / max(len(judge_results), 1)
            log.info("Judge %s/%s: avg_rating=%.2f", exp, test_file.stem, avg_rating)
