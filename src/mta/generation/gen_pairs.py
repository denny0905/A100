"""Generate TGO/SGO pairs for DistiLLM-2."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from omegaconf import DictConfig

from mta.data.spans import SpanExtractor
from mta.generation.backend import generate_texts
from mta.models.loading import load_model, load_smoke_models, load_tokenizer

log = logging.getLogger(__name__)


def run(cfg: DictConfig, experiment: str) -> None:
    """Generate TGO/SGO pairs for DistiLLM-2 training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir = Path(cfg.data.processed_dir)
    train_file = processed_dir / "dolly_train.jsonl"

    with train_file.open("r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f if line.strip()]

    out_dir = Path(cfg.output_dir) / cfg.pair_name / experiment / "gen_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [item["prompt"] for item in train_data]
    max_new = cfg.distillm2.gen_max_new_tokens

    # TGO: teacher-generated outputs (cached, generated once)
    tgo_file = out_dir / "tgo.jsonl"
    if not tgo_file.exists():
        log.info("Generating TGO (teacher outputs)...")
        if cfg.smoke:
            tokenizer, teacher, _ = load_smoke_models(cfg)
            max_new = cfg.smoke_cfg.max_new_tokens
        else:
            tokenizer = load_tokenizer(cfg.teacher.model_name)
            teacher_path = cfg.teacher.path or cfg.teacher.model_name
            teacher = load_model(teacher_path, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)

        teacher.to(device)
        teacher.eval()

        tgo_texts = generate_texts(
            teacher, tokenizer, prompts,
            max_new_tokens=max_new,
            batch_size=cfg.distillm2.gen_batch_size,
            device=device,
        )

        with tgo_file.open("w", encoding="utf-8") as f:
            for prompt, gen in zip(prompts, tgo_texts):
                f.write(json.dumps({"prompt": prompt, "output": gen}, ensure_ascii=False) + "\n")

        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # SGO: student-generated outputs (regenerated each epoch)
    sgo_file = out_dir / "sgo.jsonl"
    log.info("Generating SGO (student outputs)...")

    if cfg.smoke:
        tokenizer, _, student = load_smoke_models(cfg)
        max_new = cfg.smoke_cfg.max_new_tokens
    else:
        tokenizer = load_tokenizer(cfg.student.model_name)
        # Load latest student checkpoint if available
        ckpt_dir = Path(cfg.output_dir) / cfg.pair_name / experiment / "last"
        if ckpt_dir.exists() and (ckpt_dir / "config.json").exists():
            from transformers import AutoModelForCausalLM
            student = AutoModelForCausalLM.from_pretrained(ckpt_dir)
        else:
            student = load_model(cfg.student.model_name, dtype=torch.float32)

    student.to(device)
    student.eval()

    sgo_texts = generate_texts(
        student, tokenizer, prompts,
        max_new_tokens=max_new,
        batch_size=cfg.distillm2.gen_batch_size,
        device=device,
    )

    with sgo_file.open("w", encoding="utf-8") as f:
        for prompt, gen in zip(prompts, sgo_texts):
            f.write(json.dumps({"prompt": prompt, "output": gen}, ensure_ascii=False) + "\n")

    del student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Extract spans for generated texts if MTA is enabled
    if "+mta" in experiment:
        log.info("Extracting spans for generated pairs...")
        extractor = SpanExtractor(
            tokenizer=tokenizer,
            spacy_model=cfg.spans.spacy_model,
            max_vp_words=cfg.spans.max_vp_words,
            max_spans=cfg.spans.max_spans,
        )

        for name, gen_file in [("tgo", tgo_file), ("sgo", sgo_file)]:
            spans_file = out_dir / f"{name}_spans.pt"
            with gen_file.open("r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f if line.strip()]
            texts = [item["prompt"] + item["output"] for item in items]
            spans = extractor.extract_batch(texts, max_len=cfg.data.max_len)
            tensor_spans = [extractor.to_tensors(s) for s in spans]
            torch.save(tensor_spans, spans_file)
            log.info("Saved spans for %s: %s", name, spans_file)
