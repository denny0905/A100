"""Generation backend: vLLM if available, else HF generate."""

from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger(__name__)


def has_vllm() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def generate_texts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    do_sample: bool = True,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> list[str]:
    """Generate text completions for a list of prompts."""
    if has_vllm() and device and device.type == "cuda":
        return _generate_vllm(model, tokenizer, prompts, max_new_tokens, temperature, top_p)
    return _generate_hf(
        model, tokenizer, prompts, max_new_tokens, temperature, top_p, top_k, do_sample, batch_size, device,
    )


def _generate_hf(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    batch_size: int,
    device: torch.device | None,
) -> list[str]:
    """Generate using HuggingFace transformers."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j in range(gen.shape[0]):
            text = tokenizer.decode(gen[j, prompt_len:], skip_special_tokens=True)
            results.append(text)

    return results


def _generate_vllm(
    model_path: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Generate using vLLM."""
    from vllm import LLM, SamplingParams

    if isinstance(model_path, str):
        llm = LLM(model=model_path)
    else:
        log.warning("vLLM requires a model path string, falling back to HF generate")
        return _generate_hf(model_path, tokenizer, prompts, max_new_tokens, temperature, top_p, 0, True, 8, None)

    params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]
