"""Generic training loop with checkpoint, resume, and best-checkpoint selection."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from mta.data.dataset import InstructionCollator, InstructionDataset
from mta.losses.distillm import DistillmLoss, ReplayBuffer
from mta.losses.distillm2 import Distillm2Loss
from mta.losses.divergences import kl_divergence
from mta.losses.fdd import FDDLoss
from mta.losses.mta import MTALoss
from mta.models.layers import LayerPair, build_layer_pairs
from mta.models.loading import (
    apply_lora,
    get_hidden_dim,
    get_n_layers,
    load_model,
    load_smoke_models,
    load_tokenizer,
)
from mta.utils.seed import set_seed

log = logging.getLogger(__name__)


def _needs_hidden_states(experiment: str) -> bool:
    return "mta" in experiment or "fdd" in experiment


def _needs_mta(experiment: str) -> bool:
    return "+mta" in experiment


def _base_method(experiment: str) -> str:
    return experiment.replace("+mta", "")


def _build_optimizer(
    student: nn.Module,
    projector_params: list[nn.Parameter],
    cfg: DictConfig,
) -> torch.optim.Optimizer:
    student_params = [p for p in student.parameters() if p.requires_grad]
    param_groups = [
        {"params": student_params, "lr": cfg.train.lr},
    ]
    if projector_params:
        param_groups.append({"params": projector_params, "lr": cfg.train.projector_lr})

    if cfg.train.use_8bit_optim:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                param_groups,
                weight_decay=cfg.train.weight_decay,
                betas=tuple(cfg.train.betas),
            )
        except ImportError:
            log.warning("bitsandbytes not available, falling back to AdamW")

    return torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.train.weight_decay,
        betas=tuple(cfg.train.betas),
    )


def _build_scheduler(optimizer: Any, total_steps: int, cfg: DictConfig) -> Any:
    warmup = cfg.train.warmup_steps
    if cfg.train.warmup_ratio > 0 and warmup == 0:
        warmup = int(total_steps * cfg.train.warmup_ratio)

    if cfg.train.scheduler == "cosine":
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)


def _save_checkpoint(
    path: Path,
    student: nn.Module,
    optimizer: Any,
    scheduler: Any,
    mta_loss: MTALoss | None,
    epoch: int,
    global_step: int,
    best_metric: float,
    extra: dict | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "rng_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["rng_cuda"] = torch.cuda.get_rng_state_all()
    if mta_loss is not None:
        state["mta_projectors"] = {
            k: v.state_dict() for k, v in mta_loss.projectors.items()
        }
    if extra:
        state.update(extra)
    torch.save(state, path / "checkpoint.pt")
    log.info("Saved checkpoint to %s (epoch %d, step %d)", path, epoch, global_step)


def _load_checkpoint(
    path: Path,
    student: nn.Module,
    optimizer: Any,
    scheduler: Any,
    mta_loss: MTALoss | None,
    device: torch.device,
) -> dict:
    ckpt = torch.load(path / "checkpoint.pt", map_location=device, weights_only=False)
    student.load_state_dict(ckpt["student"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    torch.random.set_rng_state(ckpt["rng_cpu"])
    if torch.cuda.is_available() and "rng_cuda" in ckpt:
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    if mta_loss is not None and "mta_projectors" in ckpt:
        for k, v in ckpt["mta_projectors"].items():
            if k in mta_loss.projectors:
                mta_loss.projectors[k].load_state_dict(v)
    log.info("Resumed from %s (epoch %d, step %d)", path, ckpt["epoch"], ckpt["global_step"])
    return ckpt


def _validate_rouge(
    student: nn.Module,
    tokenizer: Any,
    val_dataset: InstructionDataset,
    cfg: DictConfig,
    device: torch.device,
) -> float:
    """Quick ROUGE-L on validation set with greedy decoding."""
    from mta.eval.rouge import compute_rouge_l

    student.eval()
    scores = []
    collator = InstructionCollator(tokenizer, max_len=cfg.data.max_len)
    loader = DataLoader(val_dataset, batch_size=cfg.eval.batch_size, collate_fn=collator)

    max_new = cfg.eval.max_new_tokens or (cfg.eval.max_length - cfg.data.max_prompt_len)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            prompt_lens = batch["prompt_lengths"]

            for i in range(input_ids.shape[0]):
                plen = prompt_lens[i].item()
                prompt_ids = input_ids[i, :plen].unsqueeze(0)

                if cfg.train.val_do_sample:
                    gen = student.generate(
                        prompt_ids,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=cfg.eval.temperature,
                        top_p=cfg.eval.top_p,
                    )
                else:
                    gen = student.generate(prompt_ids, max_new_tokens=max_new, do_sample=False)

                pred_text = tokenizer.decode(gen[0, plen:], skip_special_tokens=True)
                ref_text = batch["full_texts"][i][len(tokenizer.decode(input_ids[i, :plen], skip_special_tokens=True)):]
                scores.append(compute_rouge_l(pred_text, ref_text))

    student.train()
    return sum(scores) / max(len(scores), 1)


def run_training(cfg: DictConfig, experiment: str) -> None:
    """Run distillation training for a given experiment."""
    logging.basicConfig(level=logging.INFO)
    set_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg.train.bf16 and device.type == "cuda"

    # Load models
    if cfg.smoke:
        tokenizer, teacher, student = load_smoke_models(cfg)
    else:
        tokenizer = load_tokenizer(cfg.student.model_name)
        student = load_model(cfg.student.model_name, dtype=torch.float32)
        teacher = load_model(
            cfg.teacher.path or cfg.teacher.model_name,
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
        )

    if cfg.lora.enabled:
        student = apply_lora(student, cfg.lora)

    teacher.eval()
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.to(device)
    student.train()

    n_student = get_n_layers(student)
    n_teacher = get_n_layers(teacher)
    d_student = get_hidden_dim(student)
    d_teacher = get_hidden_dim(teacher)

    # Setup losses
    base_method = _base_method(experiment)
    use_mta = _needs_mta(experiment)
    need_hidden = _needs_hidden_states(experiment)

    mta_loss_mod: MTALoss | None = None
    if use_mta:
        layer_pairs = build_layer_pairs(
            list(cfg.mta.word_layers),
            list(cfg.mta.phrase_layers),
            n_student, n_teacher,
        )
        mta_loss_mod = MTALoss(
            d_student=d_student,
            d_teacher=d_teacher,
            layer_pairs=layer_pairs,
            student_pool_weights=cfg.mta.student_pool_weights,
            hid_renorm=cfg.mta.hid_renorm,
            chunk_size=cfg.mta.chunk_size,
        ).to(device)

    base_loss_mod: nn.Module | None = None
    replay_buffer: ReplayBuffer | None = None
    distillm2_loss: Distillm2Loss | None = None

    if base_method == "fdd":
        fdd_layers = list(cfg.fdd.layers or (list(cfg.mta.word_layers) + list(cfg.mta.phrase_layers)))
        base_loss_mod = FDDLoss(
            layers=fdd_layers,
            n_student=n_student,
            n_teacher=n_teacher,
            apply_final_norm=cfg.fdd.apply_final_norm,
            include_final_kd=cfg.fdd.include_final_kd,
            lambda_traj=cfg.fdd.lambda_traj,
            lambda_der=cfg.fdd.lambda_der,
            lambda_kd=cfg.fdd.lambda_kd,
            chunk_size=cfg.loss.chunk_size,
        )
    elif base_method == "distillm":
        base_loss_mod = DistillmLoss(
            loss_type=cfg.distillm.loss_type,
            alpha=cfg.distillm.alpha,
            chunk_size=cfg.loss.chunk_size,
        )
        replay_buffer = ReplayBuffer(max_size=cfg.distillm.replay_buffer_size)
    elif base_method == "distillm2":
        distillm2_loss = Distillm2Loss(
            alpha_init=cfg.distillm2.alpha_init,
            beta_init=cfg.distillm2.beta_init,
            beta_final=cfg.distillm2.beta_final,
            total_steps=0,  # set after we know total_steps
            chunk_size=cfg.loss.chunk_size,
        )
    elif base_method == "sft":
        pass  # plain cross-entropy

    # Data
    processed_dir = Path(cfg.data.processed_dir)
    train_ds = InstructionDataset(
        processed_dir / "dolly_train.jsonl", tokenizer,
        max_len=cfg.data.max_len, max_prompt_len=cfg.data.max_prompt_len,
    )
    val_ds = InstructionDataset(
        processed_dir / "dolly_valid.jsonl", tokenizer,
        max_len=cfg.data.max_len, max_prompt_len=cfg.data.max_prompt_len,
    )
    collator = InstructionCollator(tokenizer, max_len=cfg.data.max_len)

    batch_size = cfg.train.batch_size
    if base_method == "distillm2":
        batch_size = 8  # Table 9

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, collate_fn=collator,
        drop_last=True,
    )

    # DistiLLM-2: load TGO/SGO data and replace train_loader
    sgo_loader = None
    tgo_spans = None
    sgo_spans = None
    if base_method == "distillm2":
        gen_dir = Path(cfg.output_dir) / cfg.pair_name / experiment / "gen_pairs"
        tgo_ds = InstructionDataset(
            gen_dir / "tgo.jsonl", tokenizer,
            max_len=cfg.data.max_len, max_prompt_len=cfg.data.max_prompt_len,
        )
        sgo_ds = InstructionDataset(
            gen_dir / "sgo.jsonl", tokenizer,
            max_len=cfg.data.max_len, max_prompt_len=cfg.data.max_prompt_len,
        )
        train_loader = DataLoader(
            tgo_ds, batch_size=batch_size, shuffle=True,
            num_workers=cfg.train.num_workers, collate_fn=collator, drop_last=True,
        )
        sgo_loader = DataLoader(
            sgo_ds, batch_size=batch_size, shuffle=True,
            num_workers=cfg.train.num_workers, collate_fn=collator, drop_last=True,
        )
        if use_mta:
            try:
                tgo_spans = torch.load(gen_dir / "tgo_spans.pt", weights_only=False)
                sgo_spans = torch.load(gen_dir / "sgo_spans.pt", weights_only=False)
            except FileNotFoundError:
                log.warning("No TGO/SGO spans found for MTA.")

    # Spans (for non-distillm2 experiments)
    span_data = None
    if use_mta and base_method != "distillm2":
        from mta.data.spans import load_cached_spans
        try:
            span_data = load_cached_spans(
                cfg.spans.cache_dir, cfg.student.model_name if not cfg.smoke else "gpt2", "dolly_train"
            )
        except FileNotFoundError:
            log.warning("No cached spans found. MTA will run without spans.")

    # Optimizer
    projector_params = list(mta_loss_mod.projectors.parameters()) if mta_loss_mod else []
    optimizer = _build_optimizer(student, projector_params, cfg)

    total_steps = len(train_loader) * cfg.train.epochs
    if cfg.smoke and cfg.smoke_cfg.train_steps:
        total_steps = cfg.smoke_cfg.train_steps
    if distillm2_loss is not None:
        distillm2_loss.total_steps = total_steps
    scheduler = _build_scheduler(optimizer, total_steps, cfg)

    # Output directory
    out_dir = Path(cfg.output_dir) / cfg.pair_name / experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    start_epoch = 0
    global_step = 0
    best_metric = -1.0

    if cfg.train.resume == "auto":
        last_ckpt = out_dir / "last"
        if last_ckpt.exists():
            ckpt = _load_checkpoint(last_ckpt, student, optimizer, scheduler, mta_loss_mod, device)
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt["global_step"]
            best_metric = ckpt.get("best_metric", -1.0)
            if replay_buffer and "replay_buffer" in ckpt:
                replay_buffer.load_state_dict(ckpt["replay_buffer"])
            if distillm2_loss is not None and "distillm2_loss" in ckpt:
                distillm2_loss.load_state_dict_extra(ckpt["distillm2_loss"])

    # Training loop
    log.info("Starting training: %s (epochs %d-%d, %d steps/epoch)",
             experiment, start_epoch, cfg.train.epochs - 1, len(train_loader))

    for epoch in range(start_epoch, cfg.train.epochs):
        student.train()
        epoch_loss = 0.0
        step_in_epoch = 0

        sgo_iter = iter(sgo_loader) if sgo_loader is not None else None

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", disable=not log.isEnabledFor(logging.INFO)):
            if cfg.smoke and cfg.smoke_cfg.train_steps and global_step >= cfg.smoke_cfg.train_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            # Forward
            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                s_out = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=need_hidden,
                )
                with torch.no_grad():
                    t_out = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=need_hidden,
                    )

            # Compute base loss
            loss = torch.tensor(0.0, device=device)

            if base_method == "sft":
                loss = nn.functional.cross_entropy(
                    s_out.logits.view(-1, s_out.logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                )
            elif base_method == "distillm2" and distillm2_loss is not None:
                # Get SGO batch
                try:
                    sgo_batch = next(sgo_iter)
                except (StopIteration, TypeError):
                    sgo_iter = iter(sgo_loader)
                    sgo_batch = next(sgo_iter)

                sgo_ids = sgo_batch["input_ids"].to(device)
                sgo_mask = sgo_batch["attention_mask"].to(device)
                sgo_resp_mask = sgo_batch["response_mask"].to(device)

                # Forward on SGO
                with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                    s_sgo_out = student(
                        input_ids=sgo_ids, attention_mask=sgo_mask,
                        output_hidden_states=need_hidden,
                    )
                    with torch.no_grad():
                        t_sgo_out = teacher(
                            input_ids=sgo_ids, attention_mask=sgo_mask,
                            output_hidden_states=need_hidden,
                        )

                distillm2_loss.update_schedule(global_step)
                cald_result = distillm2_loss(
                    t_out.logits.float(), s_out.logits.float(),
                    t_sgo_out.logits.float(), s_sgo_out.logits.float(),
                    response_mask, sgo_resp_mask,
                )
                loss = cald_result["total"]
            elif base_method == "fdd" and base_loss_mod is not None:
                fdd_result = base_loss_mod(
                    s_out.hidden_states, t_out.hidden_states,
                    student, teacher, response_mask,
                )
                loss = fdd_result["total"]
            elif base_method == "distillm" and base_loss_mod is not None:
                loss = base_loss_mod(t_out.logits, s_out.logits, response_mask)

            # MTA loss
            if use_mta and mta_loss_mod is not None:
                if base_method == "distillm2" and (tgo_spans is not None or sgo_spans is not None):
                    mta_on = cfg.distillm2.mta_on
                    mta_loss_val = torch.tensor(0.0, device=device)
                    n_mta = 0

                    if mta_on in ("both", "tgo") and tgo_spans is not None:
                        bi = list(range(
                            step_in_epoch * batch_size,
                            min((step_in_epoch + 1) * batch_size, len(tgo_spans)),
                        ))
                        if bi and max(bi) < len(tgo_spans):
                            bs = [tgo_spans[i] for i in bi]
                            mta_r = mta_loss_mod(
                                s_out.hidden_states, t_out.hidden_states,
                                attention_mask, bs,
                            )
                            mta_loss_val = mta_loss_val + cfg.mta.lambda_dsa * mta_r["dsa"]
                            mta_loss_val = mta_loss_val + cfg.mta.lambda_hid * mta_r["hid"]
                            n_mta += 1

                    if mta_on in ("both", "sgo") and sgo_spans is not None:
                        bi = list(range(
                            step_in_epoch * batch_size,
                            min((step_in_epoch + 1) * batch_size, len(sgo_spans)),
                        ))
                        if bi and max(bi) < len(sgo_spans):
                            bs = [sgo_spans[i] for i in bi]
                            mta_r = mta_loss_mod(
                                s_sgo_out.hidden_states, t_sgo_out.hidden_states,
                                sgo_mask, bs,
                            )
                            mta_loss_val = mta_loss_val + cfg.mta.lambda_dsa * mta_r["dsa"]
                            mta_loss_val = mta_loss_val + cfg.mta.lambda_hid * mta_r["hid"]
                            n_mta += 1

                    if n_mta > 1:
                        mta_loss_val = mta_loss_val / n_mta
                    loss = loss + mta_loss_val

                elif span_data is not None:
                    batch_indices = list(range(
                        step_in_epoch * batch_size,
                        min((step_in_epoch + 1) * batch_size, len(span_data))
                    ))
                    if batch_indices and max(batch_indices) < len(span_data):
                        batch_spans = [span_data[i] for i in batch_indices]
                        mta_result = mta_loss_mod(
                            s_out.hidden_states, t_out.hidden_states,
                            attention_mask, batch_spans,
                        )
                        loss = loss + cfg.mta.lambda_dsa * mta_result["dsa"]
                        loss = loss + cfg.mta.lambda_hid * mta_result["hid"]

            # Backward
            loss = loss / cfg.train.grad_accumulation
            loss.backward()

            if (global_step + 1) % cfg.train.grad_accumulation == 0:
                if cfg.train.max_grad_norm:
                    params = list(student.parameters())
                    if mta_loss_mod:
                        params += list(mta_loss_mod.projectors.parameters())
                    nn.utils.clip_grad_norm_(params, cfg.train.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * cfg.train.grad_accumulation
            global_step += 1
            step_in_epoch += 1

            if cfg.train.save_steps > 0 and global_step % cfg.train.save_steps == 0:
                extra_mid = {}
                if replay_buffer:
                    extra_mid["replay_buffer"] = replay_buffer.state_dict()
                if distillm2_loss is not None:
                    extra_mid["distillm2_loss"] = distillm2_loss.state_dict_extra()
                _save_checkpoint(
                    out_dir / "last", student, optimizer, scheduler, mta_loss_mod,
                    epoch, global_step, best_metric, extra_mid,
                )

        avg_loss = epoch_loss / max(step_in_epoch, 1)
        log.info("Epoch %d: avg_loss=%.4f", epoch, avg_loss)

        # Validation
        try:
            val_rouge = _validate_rouge(student, tokenizer, val_ds, cfg, device)
            log.info("Epoch %d: val_rouge_l=%.4f", epoch, val_rouge)
        except Exception as e:
            log.warning("Validation failed: %s", e)
            val_rouge = avg_loss * -1  # fallback: lower loss is better

        # Save checkpoints
        extra = {}
        if replay_buffer:
            extra["replay_buffer"] = replay_buffer.state_dict()
        if distillm2_loss is not None:
            extra["distillm2_loss"] = distillm2_loss.state_dict_extra()

        _save_checkpoint(
            out_dir / "last", student, optimizer, scheduler, mta_loss_mod,
            epoch, global_step, best_metric, extra,
        )

        if val_rouge > best_metric:
            best_metric = val_rouge
            _save_checkpoint(
                out_dir / "best", student, optimizer, scheduler, mta_loss_mod,
                epoch, global_step, best_metric, extra,
            )
            log.info("New best: %.4f", best_metric)

    log.info("Training complete for %s. Best metric: %.4f", experiment, best_metric)


if __name__ == "__main__":
    from mta.config import parse_cli

    logging.basicConfig(level=logging.INFO)
    cfg, flags = parse_cli()
    exp = cfg.experiment
    run_training(cfg, exp)
