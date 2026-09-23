"""Smoke test: full pipeline runs on CPU with tiny random models."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mta.config import load_config
from mta.pipeline import run_pipeline


@pytest.fixture(autouse=True)
def clean_artifacts(tmp_path):
    """Redirect all output to tmp_path so tests don't pollute the workspace."""
    yield
    for d in ("data/processed", "data/spans", "data/raw"):
        p = Path(d)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


class TestSmokePipeline:
    @pytest.mark.timeout(600)
    def test_full_pipeline(self, tmp_path):
        """All 7 experiments run end-to-end on CPU with tiny models."""
        cfg = load_config("configs/smoke.yaml", overrides=[
            f"output_dir={tmp_path / 'runs'}",
        ])

        run_pipeline(cfg, {
            "only": None,
            "from_stage": None,
            "experiments": None,
        })

        runs_dir = tmp_path / "runs" / "smoke"

        # SFT teacher checkpoint exists
        assert (runs_dir / "sft_teacher" / "best").exists()

        # All 7 experiments have training output
        for exp in ["sft", "fdd", "fdd+mta", "distillm", "distillm+mta", "distillm2", "distillm2+mta"]:
            last_dir = runs_dir / exp / "last"
            assert last_dir.exists(), f"Missing last checkpoint for {exp}"
            assert (last_dir / "checkpoint.pt").exists(), f"Missing checkpoint.pt for {exp}"

        # Eval results exist for all experiments and test sets
        results_dir = Path("results") / "smoke"
        assert results_dir.exists()
        for exp in ["sft", "fdd", "fdd+mta", "distillm", "distillm+mta", "distillm2", "distillm2+mta"]:
            for ts in ["dolly_test", "selfinst_test", "vicuna_test", "sni_test"]:
                result_file = results_dir / exp / f"{ts}_seed42.jsonl"
                assert result_file.exists(), f"Missing eval result: {result_file}"
                with result_file.open() as f:
                    items = [json.loads(line) for line in f if line.strip()]
                assert len(items) > 0, f"Empty eval result: {result_file}"
                assert "rouge_l" in items[0], f"Missing rouge_l key in {result_file}"

        # Report exists
        assert (results_dir / "summary.md").exists()
        assert (results_dir / "summary.csv").exists()

    @pytest.mark.timeout(600)
    def test_idempotent_rerun(self, tmp_path):
        """Running the pipeline twice skips already-done stages."""
        cfg = load_config("configs/smoke.yaml", overrides=[
            f"output_dir={tmp_path / 'runs'}",
        ])
        flags = {"only": None, "from_stage": None, "experiments": ["sft"]}

        run_pipeline(cfg, flags)
        # Second run should skip all stages
        run_pipeline(cfg, flags)

        runs_dir = tmp_path / "runs" / "smoke"
        assert (runs_dir / "sft" / "last" / "checkpoint.pt").exists()
