"""Download raw data from HF Hub or original sources."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from omegaconf import DictConfig

log = logging.getLogger(__name__)

MINILLM_DATA_URLS = {
    "dolly": "https://huggingface.co/datasets/gpt2-xl-minillm/dolly",
    "selfinst": "https://huggingface.co/datasets/gpt2-xl-minillm/self_inst",
    "vicuna": "https://huggingface.co/datasets/gpt2-xl-minillm/vicuna",
    "sni": "https://huggingface.co/datasets/gpt2-xl-minillm/sinst",
}


def _download_from_hub(repo_id: str, dest: Path) -> None:
    """Download a dataset repo from HF Hub."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        token=os.environ.get("HF_TOKEN"),
    )


def _download_dolly_raw(dest: Path) -> None:
    """Download Dolly from the original HF dataset."""
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    out_file = dest / "dolly_raw.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for item in ds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log.info("Downloaded Dolly: %d samples -> %s", len(ds), out_file)


def run(cfg: DictConfig) -> None:
    """Download raw Dolly data."""
    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if cfg.smoke:
        log.info("Smoke mode: skipping data download (will use fixtures or generate synthetic)")
        return

    if cfg.data.hub_repo:
        log.info("Downloading from user hub repo: %s", cfg.data.hub_repo)
        _download_from_hub(cfg.data.hub_repo, raw_dir)
        return

    dolly_file = raw_dir / "dolly_raw.jsonl"
    if dolly_file.exists():
        log.info("Raw Dolly data already exists at %s", dolly_file)
        return

    log.info("Downloading raw Dolly dataset...")
    _download_dolly_raw(raw_dir)

    if not dolly_file.exists():
        raise RuntimeError("Failed to download Dolly data.")


def push_to_hub(processed_dir: str, repo_id: str) -> None:
    """Push processed data to user's HF Hub repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=processed_dir,
        repo_id=repo_id,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    log.info("Pushed processed data to %s", repo_id)


if __name__ == "__main__":
    import argparse

    from mta.config import parse_cli

    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args, _ = parser.parse_known_args()

    cfg, _ = parse_cli()

    if args.push:
        if cfg.data.hub_repo:
            push_to_hub(cfg.data.processed_dir, cfg.data.hub_repo)
        else:
            print("Set data.hub_repo in config to push.")
    else:
        run(cfg)
