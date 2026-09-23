"""Configuration schema and loading for MTA experiments."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Structured config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    hub_repo: Optional[str] = None
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    prompt_template: Optional[str] = None  # path override; else built-in
    max_prompt_len: int = 256
    max_len: int = 512
    split_seed: int = 42
    valid_size: int = 1000
    test_size: int = 500


@dataclass
class SpansConfig:
    spacy_model: str = "en_core_web_sm"
    cache_dir: str = "data/spans"
    max_vp_words: int = 12
    allow_overlap: bool = False
    max_spans: int = 256
    batch_size: int = 64
    n_process: int = 1


@dataclass
class TeacherConfig:
    model_name: str = "gpt2-xl"
    path: Optional[str] = None  # skip SFT if provided
    push_repo: Optional[str] = None
    lr: float = 5e-5
    epochs: int = 10
    batch_size: int = 16
    grad_accumulation: int = 1
    max_grad_norm: float = 1.0


@dataclass
class StudentConfig:
    model_name: str = "gpt2"
    student_init: str = "pretrained"  # "pretrained" | "sft"


@dataclass
class MtaConfig:
    word_layers: list[int] = field(default_factory=lambda: [6])
    phrase_layers: list[int] = field(default_factory=lambda: [9, 12])
    lambda_dsa: float = 2.0
    lambda_hid: float = 0.2
    student_pool_weights: str = "teacher"  # "teacher" | "student"
    hid_renorm: bool = False
    chunk_size: int = 0  # 0 = no chunking for token weight matrix


@dataclass
class FddConfig:
    layers: Optional[list[int]] = None  # defaults to mta word+phrase layers
    apply_final_norm: bool = True
    include_final_kd: bool = True
    lambda_traj: float = 1.0
    lambda_der: float = 1.0
    lambda_kd: float = 1.0


@dataclass
class DistillmConfig:
    loss_type: str = "srkl"  # "skl" | "srkl"
    alpha: float = 0.1
    sgo_max_new_tokens: int = 256
    replay_buffer_size: int = 512


@dataclass
class Distillm2Config:
    alpha_init: float = 0.1
    beta_init: float = 0.0
    beta_final: float = 0.5
    mta_on: str = "both"  # "both" | "tgo" | "sgo"
    gen_batch_size: int = 32
    gen_max_new_tokens: int = 256


@dataclass
class TrainConfig:
    lr: float = 1e-4
    projector_lr: float = 5e-4
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    scheduler: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    epochs: int = 5
    batch_size: int = 16
    grad_accumulation: int = 1
    grad_checkpointing: bool = False
    num_workers: int = 0
    seed: int = 42
    bf16: bool = True  # auto-disabled on CPU
    val_do_sample: bool = False
    resume: str = "auto"  # "auto" | "none" | path
    use_8bit_optim: bool = False  # requires bitsandbytes


@dataclass
class LoraConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 64
    dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )


@dataclass
class LossConfig:
    chunk_size: int = 512  # chunk along sequence for KL computation


@dataclass
class EvalConfig:
    seeds: list[int] = field(default_factory=lambda: [10, 20, 30, 40, 50])
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    max_length: int = 512
    max_new_tokens: Optional[int] = None
    batch_size: int = 8
    test_sets: list[str] = field(
        default_factory=lambda: [
            "dolly_test",
            "selfinst_test",
            "vicuna_test",
            "sni_test",
        ]
    )


@dataclass
class JudgeConfig:
    enabled: bool = False
    model: str = "gpt-4o-mini"


@dataclass
class ResultsConfig:
    push_repo: Optional[str] = None


@dataclass
class SmokeConfig:
    teacher_n_layer: int = 8
    teacher_n_embd: int = 128
    teacher_n_head: int = 4
    student_n_layer: int = 4
    student_n_embd: int = 64
    student_n_head: int = 4
    train_samples: int = 32
    valid_samples: int = 16
    test_samples: int = 8
    max_len: int = 128
    max_new_tokens: int = 16
    train_steps: Optional[int] = 4
    sft_steps: Optional[int] = 2


@dataclass
class MTAExperimentConfig:
    pair_name: str = "gpt2"
    experiment: str = "sft"  # set at runtime or by pipeline
    experiments: list[str] = field(
        default_factory=lambda: [
            "sft",
            "fdd",
            "fdd+mta",
            "distillm",
            "distillm+mta",
            "distillm2",
            "distillm2+mta",
        ]
    )

    data: DataConfig = field(default_factory=DataConfig)
    spans: SpansConfig = field(default_factory=SpansConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    mta: MtaConfig = field(default_factory=MtaConfig)
    fdd: FddConfig = field(default_factory=FddConfig)
    distillm: DistillmConfig = field(default_factory=DistillmConfig)
    distillm2: Distillm2Config = field(default_factory=Distillm2Config)
    train: TrainConfig = field(default_factory=TrainConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    results: ResultsConfig = field(default_factory=ResultsConfig)

    smoke: bool = False
    smoke_cfg: SmokeConfig = field(default_factory=SmokeConfig)

    output_dir: str = "runs"
    log_wandb: bool = False


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load config from YAML file, merge with structured defaults, apply CLI overrides."""
    schema = OmegaConf.structured(MTAExperimentConfig)

    if config_path is not None:
        file_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(schema, file_cfg)
    else:
        cfg = schema

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    OmegaConf.resolve(cfg)
    return cfg


def parse_cli() -> tuple[DictConfig, dict[str, Any]]:
    """Parse CLI arguments: --config PATH, --smoke, --only, --from, --experiments, and key=value overrides."""
    import argparse

    parser = argparse.ArgumentParser(description="MTA experiment pipeline")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated stages to run")
    parser.add_argument("--from", dest="from_stage", type=str, default=None, help="Start from this stage")
    parser.add_argument(
        "--experiments", type=str, default=None,
        help="Comma-separated experiment names to run",
    )

    args, unknown = parser.parse_known_args()

    overrides = []
    for u in unknown:
        if "=" in u:
            overrides.append(u)

    if args.smoke:
        overrides.append("smoke=true")

    cfg = load_config(args.config, overrides if overrides else None)

    cli_flags = {
        "only": args.only.split(",") if args.only else None,
        "from_stage": args.from_stage,
        "experiments": args.experiments.split(",") if args.experiments else None,
    }

    return cfg, cli_flags


if __name__ == "__main__":
    cfg, flags = parse_cli()
    print(OmegaConf.to_yaml(cfg))
    print(f"CLI flags: {flags}")
