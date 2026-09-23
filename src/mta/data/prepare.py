"""Format raw data into processed JSONL splits with prompt templates."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

log = logging.getLogger(__name__)

TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_prompt(instruction: str, input_text: str = "") -> str:
    """Apply the MiniLLM prompt template."""
    if input_text and input_text.strip():
        return TEMPLATE_WITH_INPUT.format(instruction=instruction, input=input_text)
    return TEMPLATE_NO_INPUT.format(instruction=instruction)


def _load_raw_dolly(raw_dir: Path) -> list[dict[str, Any]]:
    """Load raw Dolly data from JSONL."""
    dolly_file = raw_dir / "dolly_raw.jsonl"
    if not dolly_file.exists():
        return []
    with dolly_file.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _process_dolly_item(item: dict[str, Any]) -> dict[str, str]:
    instruction = item.get("instruction", "")
    input_text = item.get("context", "")
    output = item.get("response", "")
    return {
        "prompt": format_prompt(instruction, input_text),
        "output": output,
    }


def _split_dolly(
    data: list[dict[str, Any]],
    seed: int,
    valid_size: int,
    test_size: int,
) -> dict[str, list[dict[str, str]]]:
    """Split Dolly into train/valid/test."""
    processed = [_process_dolly_item(item) for item in data]
    for i, item in enumerate(processed):
        item["id"] = f"dolly_{i}"

    rng = random.Random(seed)
    indices = list(range(len(processed)))
    rng.shuffle(indices)

    test_idx = set(indices[:test_size])
    valid_idx = set(indices[test_size : test_size + valid_size])

    splits: dict[str, list[dict[str, str]]] = {
        "dolly_train": [],
        "dolly_valid": [],
        "dolly_test": [],
    }
    for i, item in enumerate(processed):
        if i in test_idx:
            splits["dolly_test"].append(item)
        elif i in valid_idx:
            splits["dolly_valid"].append(item)
        else:
            splits["dolly_train"].append(item)

    return splits


def _write_split(data: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log.info("Wrote %d samples to %s", len(data), path)


def _generate_smoke_data(cfg: DictConfig, out_dir: Path) -> None:
    """Generate tiny synthetic data for smoke testing."""
    smoke = cfg.smoke_cfg
    templates = [
        ("Summarize the following text.", "The quick brown fox jumps over the lazy dog.", "A fox jumps over a dog."),
        ("Translate to French.", "Hello world", "Bonjour le monde"),
        ("What is machine learning?", "", "Machine learning is a subset of AI."),
        ("Write a poem about nature.", "", "Trees sway in the gentle breeze."),
        ("Explain photosynthesis.", "", "Plants convert sunlight to energy."),
        ("List three colors.", "", "Red, blue, and green."),
        ("Define gravity.", "", "Gravity is a force of attraction."),
        ("What is Python?", "", "Python is a programming language."),
    ]

    def make_samples(n: int, prefix: str) -> list[dict[str, str]]:
        samples = []
        for i in range(n):
            t = templates[i % len(templates)]
            samples.append({
                "id": f"{prefix}_{i}",
                "prompt": format_prompt(t[0], t[1]),
                "output": t[2],
            })
        return samples

    splits = {
        "dolly_train": make_samples(smoke.train_samples, "train"),
        "dolly_valid": make_samples(smoke.valid_samples, "valid"),
        "dolly_test": make_samples(smoke.test_samples, "test"),
        "selfinst_test": make_samples(smoke.test_samples, "selfinst"),
        "vicuna_test": make_samples(smoke.test_samples, "vicuna"),
        "sni_test": make_samples(smoke.test_samples, "sni"),
    }

    for name, data in splits.items():
        _write_split(data, out_dir / f"{name}.jsonl")


def run(cfg: DictConfig) -> None:
    """Prepare processed data splits."""
    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.smoke:
        log.info("Smoke mode: generating synthetic data")
        _generate_smoke_data(cfg, out_dir)
        return

    # Check if processed data already exists
    expected = ["dolly_train.jsonl", "dolly_valid.jsonl", "dolly_test.jsonl"]
    if all((out_dir / f).exists() for f in expected):
        log.info("Processed data already exists at %s", out_dir)
        return

    # Try to find pre-processed data from downloads
    raw_dir = Path(cfg.data.raw_dir)

    # Process raw Dolly
    dolly_data = _load_raw_dolly(raw_dir)
    if dolly_data:
        splits = _split_dolly(
            dolly_data,
            seed=cfg.data.split_seed,
            valid_size=cfg.data.valid_size,
            test_size=cfg.data.test_size,
        )
        for name, data in splits.items():
            _write_split(data, out_dir / f"{name}.jsonl")
    else:
        log.warning(
            "No raw Dolly data found at %s. Run download stage first or "
            "use smoke mode.",
            raw_dir,
        )


if __name__ == "__main__":
    from mta.config import parse_cli

    cfg, _ = parse_cli()
    run(cfg)
