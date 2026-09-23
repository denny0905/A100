"""Test: train 4 steps continuous vs train 2 + save + resume + 2 → weights match."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

from mta.data.dataset import InstructionCollator, InstructionDataset
from mta.data.prepare import format_prompt
from mta.train.trainer import _build_optimizer, _build_scheduler, _save_checkpoint, _load_checkpoint
from mta.utils.seed import set_seed


def _create_test_data(tmp_dir: Path, n: int = 4) -> Path:
    """Create minimal JSONL test data. Default n=4 = one batch, so every
    training step sees the same data regardless of epoch/resume point."""
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "dolly_train.jsonl"
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "id": str(i),
                "prompt": format_prompt(f"Task {i}"),
                "output": f"Response {i}",
            }) + "\n")
    valid_path = data_dir / "dolly_valid.jsonl"
    with valid_path.open("w") as f:
        for i in range(4):
            f.write(json.dumps({
                "id": f"v{i}",
                "prompt": format_prompt(f"Valid {i}"),
                "output": f"Valid response {i}",
            }) + "\n")
    return data_dir


def _make_model():
    config = GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=50257)
    return GPT2LMHeadModel(config)


def _train_n_steps(
    model, tokenizer, train_ds, n_steps, optimizer=None, scheduler=None, start_step=0, seed=42,
):
    """Train for exactly n_steps, returning model state dict and loss history."""
    if seed is not None:
        set_seed(seed)

    collator = InstructionCollator(tokenizer, max_len=64)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=False, collate_fn=collator, drop_last=True,
    )

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)

    model.train()
    losses = []
    step = start_step
    data_iter = iter(loader)

    while step < start_step + n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = torch.nn.functional.cross_entropy(
            out.logits.view(-1, out.logits.shape[-1]),
            batch["labels"].view(-1),
            ignore_index=-100,
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        step += 1

    return model.state_dict(), losses, optimizer, scheduler


class TestResume:
    def test_continuous_vs_resumed(self, tmp_path):
        """4 continuous steps should match 2+save+resume+2 steps."""
        data_dir = _create_test_data(tmp_path)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_ds = InstructionDataset(
            data_dir / "dolly_train.jsonl", tokenizer, max_len=64, max_prompt_len=32,
        )

        # --- Continuous run: 4 steps ---
        set_seed(42)
        model_cont = _make_model()
        state_cont, losses_cont, _, _ = _train_n_steps(
            model_cont, tokenizer, train_ds, n_steps=4, seed=42,
        )

        # --- Resumed run: 2 steps + save + resume + 2 steps ---
        set_seed(42)
        model_res = _make_model()
        state_2, losses_2, opt_2, sched_2 = _train_n_steps(
            model_res, tokenizer, train_ds, n_steps=2, seed=42,
        )

        # Save checkpoint
        ckpt_dir = tmp_path / "ckpt"
        _save_checkpoint(
            ckpt_dir, model_res, opt_2, sched_2,
            mta_loss=None, epoch=0, global_step=2, best_metric=-1.0,
        )

        # Load into fresh model
        model_res2 = _make_model()
        opt_res2 = torch.optim.AdamW(model_res2.parameters(), lr=1e-3)
        sched_res2 = torch.optim.lr_scheduler.LambdaLR(opt_res2, lambda s: 1.0)

        ckpt = _load_checkpoint(ckpt_dir, model_res2, opt_res2, sched_res2, None, torch.device("cpu"))

        # Continue for 2 more steps (seed=None to preserve checkpoint RNG state)
        state_res, losses_res, _, _ = _train_n_steps(
            model_res2, tokenizer, train_ds, n_steps=2,
            optimizer=opt_res2, scheduler=sched_res2, start_step=2, seed=None,
        )

        # Compare final weights
        for key in state_cont:
            assert torch.allclose(
                state_cont[key].float(), state_res[key].float(), atol=1e-4,
            ), f"Weight mismatch at {key}: max diff = {(state_cont[key].float() - state_res[key].float()).abs().max()}"

    def test_checkpoint_saves_loads(self, tmp_path):
        """Basic checkpoint save/load roundtrip."""
        model = _make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

        ckpt_dir = tmp_path / "ckpt"
        _save_checkpoint(ckpt_dir, model, opt, sched, None, epoch=3, global_step=100, best_metric=42.0)

        model2 = _make_model()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 1.0)

        ckpt = _load_checkpoint(ckpt_dir, model2, opt2, sched2, None, torch.device("cpu"))
        assert ckpt["epoch"] == 3
        assert ckpt["global_step"] == 100
        assert ckpt["best_metric"] == 42.0
