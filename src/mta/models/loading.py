"""Load teacher/student models with optional LoRA, grad checkpointing, dtype."""

from __future__ import annotations

import logging
from typing import Any

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

log = logging.getLogger(__name__)


def load_tokenizer(model_name: str) -> Any:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_model(
    model_name: str,
    dtype: torch.dtype = torch.float32,
    gradient_checkpointing: bool = False,
) -> Any:
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def load_smoke_models(cfg: DictConfig) -> tuple[Any, Any, Any]:
    """Create tiny random GPT-2 models for smoke testing."""
    smoke = cfg.smoke_cfg
    tokenizer = load_tokenizer("gpt2")

    t_config = GPT2Config(
        n_layer=smoke.teacher_n_layer,
        n_embd=smoke.teacher_n_embd,
        n_head=smoke.teacher_n_head,
        vocab_size=tokenizer.vocab_size,
    )
    teacher = GPT2LMHeadModel(t_config)

    s_config = GPT2Config(
        n_layer=smoke.student_n_layer,
        n_embd=smoke.student_n_embd,
        n_head=smoke.student_n_head,
        vocab_size=tokenizer.vocab_size,
    )
    student = GPT2LMHeadModel(s_config)

    return tokenizer, teacher, student


def apply_lora(model: Any, lora_cfg: DictConfig) -> Any:
    """Wrap model with LoRA using peft."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=list(lora_cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    log.info("Applied LoRA: r=%d, alpha=%d", lora_cfg.r, lora_cfg.alpha)
    return model


def get_n_layers(model: Any) -> int:
    """Get number of transformer layers from model config."""
    config = model.config
    for attr in ("n_layer", "num_hidden_layers", "num_layers"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError("Cannot determine number of layers from model config")


def get_hidden_dim(model: Any) -> int:
    """Get hidden dimension from model config."""
    config = model.config
    for attr in ("n_embd", "hidden_size", "d_model"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError("Cannot determine hidden dimension from model config")
