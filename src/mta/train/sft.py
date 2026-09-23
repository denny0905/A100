"""SFT for teacher and student models."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from mta.data.dataset import InstructionCollator, InstructionDataset
from mta.models.loading import load_model, load_smoke_models, load_tokenizer
from mta.utils.seed import set_seed

log = logging.getLogger(__name__)


def _train_sft(
    model: nn.Module,
    tokenizer,
    train_ds: InstructionDataset,
    lr: float,
    epochs: int,
    batch_size: int,
    max_grad_norm: float,
    device: torch.device,
    out_dir: Path,
    use_bf16: bool = False,
    max_steps: int | None = None,
) -> None:
    collator = InstructionCollator(tokenizer, max_len=train_ds.max_len)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    if max_steps:
        total_steps = max_steps

    def lr_lambda(step: int) -> float:
        return 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0.0
        n_steps = 0
        for batch in tqdm(loader, desc=f"SFT Epoch {epoch}"):
            if max_steps and global_step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = nn.functional.cross_entropy(
                    out.logits.view(-1, out.logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_steps += 1
            global_step += 1

        avg = total_loss / max(n_steps, 1)
        log.info("SFT Epoch %d: loss=%.4f", epoch, avg)

        if avg < best_loss:
            best_loss = avg
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir / "best")
            tokenizer.save_pretrained(out_dir / "best")

    model.save_pretrained(out_dir / "last")
    tokenizer.save_pretrained(out_dir / "last")


def train_sft_teacher(cfg: DictConfig) -> None:
    """SFT the teacher model."""
    if cfg.teacher.path:
        log.info("Teacher checkpoint at %s, skipping SFT.", cfg.teacher.path)
        return

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg.train.bf16 and device.type == "cuda"

    if cfg.smoke:
        tokenizer, teacher, _ = load_smoke_models(cfg)
    else:
        tokenizer = load_tokenizer(cfg.teacher.model_name)
        teacher = load_model(cfg.teacher.model_name, dtype=torch.float32)

    teacher.to(device)

    processed_dir = Path(cfg.data.processed_dir)
    train_ds = InstructionDataset(
        processed_dir / "dolly_train.jsonl", tokenizer,
        max_len=cfg.data.max_len, max_prompt_len=cfg.data.max_prompt_len,
    )

    out_dir = Path(cfg.output_dir) / cfg.pair_name / "sft_teacher"
    max_steps = cfg.smoke_cfg.sft_steps if cfg.smoke else None

    _train_sft(
        teacher, tokenizer, train_ds,
        lr=cfg.teacher.lr,
        epochs=cfg.teacher.epochs,
        batch_size=cfg.teacher.batch_size,
        max_grad_norm=cfg.teacher.max_grad_norm,
        device=device,
        out_dir=out_dir,
        use_bf16=use_bf16,
        max_steps=max_steps,
    )

    if cfg.teacher.push_repo:
        from huggingface_hub import HfApi
        import os

        api = HfApi()
        api.upload_folder(
            folder_path=str(out_dir / "best"),
            repo_id=cfg.teacher.push_repo,
            token=os.environ.get("HF_TOKEN"),
        )
        log.info("Pushed SFT teacher to %s", cfg.teacher.push_repo)


def train_sft_student(cfg: DictConfig) -> None:
    """SFT the student model (used when student_init='sft')."""
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.smoke:
        tokenizer, _, student = load_smoke_models(cfg)
    else:
        tokenizer = load_tokenizer(cfg.student.model_name)
        student = load_model(cfg.student.model_name, dtype=torch.float32)

    student.to(device)

    processed_dir = Path(cfg.data.processed_dir)
    train_ds = InstructionDataset(
        processed_dir / "dolly_train.jsonl", tokenizer,
        max_len=cfg.data.max_len, max_prompt_len=cfg.data.max_prompt_len,
    )

    out_dir = Path(cfg.output_dir) / cfg.pair_name / "sft_student"
    max_steps = cfg.smoke_cfg.sft_steps if cfg.smoke else None

    _train_sft(
        student, tokenizer, train_ds,
        lr=cfg.train.lr,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        max_grad_norm=cfg.train.max_grad_norm,
        device=device,
        out_dir=out_dir,
        max_steps=max_steps,
    )
