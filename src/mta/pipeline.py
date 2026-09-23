"""Orchestrator: runs stages in order with DONE markers for idempotency."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from mta.config import parse_cli

log = logging.getLogger(__name__)

STAGES_ORDER = [
    "data",
    "spans",
    "sft_teacher",
    "sft_student",
    # per-experiment stages are inserted dynamically
    "report",
]

PER_EXPERIMENT_STAGES = ["gen_pairs", "train", "eval"]


def _cfg_hash(cfg: DictConfig, keys: list[str] | None = None) -> str:
    """Hash relevant config keys to detect config changes."""
    if keys:
        sub = {}
        for k in keys:
            val = OmegaConf.select(cfg, k)
            if val is None:
                continue
            if isinstance(val, (DictConfig,)):
                sub[k] = OmegaConf.to_container(val, resolve=True)
            else:
                sub[k] = val
    else:
        sub = OmegaConf.to_container(cfg, resolve=True)
    return hashlib.md5(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _done_path(stage_dir: Path) -> Path:
    return stage_dir / "DONE"


def _is_done(stage_dir: Path, expected_hash: str) -> bool:
    done = _done_path(stage_dir)
    if not done.exists():
        return False
    stored = done.read_text().strip()
    return stored == expected_hash


def _mark_done(stage_dir: Path, cfg_hash: str) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    _done_path(stage_dir).write_text(cfg_hash)


def _build_stage_list(
    cfg: DictConfig,
    only: list[str] | None,
    from_stage: str | None,
    experiments: list[str] | None,
) -> list[tuple[str, str | None]]:
    """Build ordered list of (stage_name, experiment_name_or_None)."""
    exp_list = experiments or list(cfg.experiments)

    full: list[tuple[str, str | None]] = []
    for s in STAGES_ORDER:
        if s == "report":
            full.append((s, None))
            continue
        full.append((s, None))
        if s == "sft_student":
            for exp in exp_list:
                needs_gen = exp.startswith("distillm2")
                for ps in PER_EXPERIMENT_STAGES:
                    if ps == "gen_pairs" and not needs_gen:
                        continue
                    full.append((ps, exp))

    if from_stage:
        idx = next(
            (i for i, (s, _) in enumerate(full) if s == from_stage), 0
        )
        full = full[idx:]

    if only:
        full = [(s, e) for s, e in full if s in only]

    return full


def run_stage(stage: str, experiment: str | None, cfg: DictConfig) -> None:
    """Dispatch a single stage."""
    log.info("=== Stage: %s %s ===", stage, f"[{experiment}]" if experiment else "")

    if stage == "data":
        from mta.data.download import run as download
        from mta.data.prepare import run as prepare
        download(cfg)
        prepare(cfg)
    elif stage == "spans":
        from mta.data.spans import precompute_all
        precompute_all(cfg)
    elif stage == "sft_teacher":
        from mta.train.sft import train_sft_teacher
        train_sft_teacher(cfg)
    elif stage == "sft_student":
        if cfg.student.student_init == "sft":
            from mta.train.sft import train_sft_student
            train_sft_student(cfg)
        else:
            log.info("Skipping SFT student (student_init=pretrained)")
    elif stage == "gen_pairs":
        from mta.generation.gen_pairs import run as gen
        gen(cfg, experiment)
    elif stage == "train":
        from mta.train.trainer import run_training
        run_training(cfg, experiment)
    elif stage == "eval":
        from mta.eval.evaluate import run_eval
        run_eval(cfg, experiment)
    elif stage == "report":
        from mta.eval.report import generate_report
        generate_report(cfg)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def run_pipeline(cfg: DictConfig, cli_flags: dict) -> None:
    """Run the full pipeline."""
    stages = _build_stage_list(
        cfg,
        only=cli_flags.get("only"),
        from_stage=cli_flags.get("from_stage"),
        experiments=cli_flags.get("experiments"),
    )

    output = Path(cfg.output_dir) / cfg.pair_name
    output.mkdir(parents=True, exist_ok=True)

    for stage, exp in stages:
        suffix = f"_{exp}" if exp else ""
        stage_dir = output / f".stage_{stage}{suffix}"
        h = _cfg_hash(cfg, [stage] if not exp else [stage, "experiment"])

        if _is_done(stage_dir, h):
            log.info("Stage %s%s already done, skipping.", stage, f" [{exp}]" if exp else "")
            continue

        if exp:
            cfg_mut = OmegaConf.to_container(cfg, resolve=True)
            cfg_mut["experiment"] = exp
            stage_cfg = OmegaConf.create(cfg_mut)
        else:
            stage_cfg = cfg

        run_stage(stage, exp, stage_cfg)
        _mark_done(stage_dir, h)

    log.info("Pipeline complete.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    cfg, flags = parse_cli()
    run_pipeline(cfg, flags)


if __name__ == "__main__":
    main()
