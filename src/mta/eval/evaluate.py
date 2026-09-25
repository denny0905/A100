"""Run evaluation: generate outputs and compute ROUGE-L."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mta.eval.rouge import compute_rouge_l
from mta.models.loading import load_smoke_models, load_tokenizer
from mta.utils.seed import set_seed

log = logging.getLogger(__name__)


def _load_test_data(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(cfg: DictConfig, experiment: str) -> None:
    """Evaluate a trained model on all test sets with multiple seeds."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_dir = Path(cfg.output_dir) / cfg.pair_name / experiment / "best"
    if not model_dir.exists():
        model_dir = Path(cfg.output_dir) / cfg.pair_name / experiment / "last"

    if cfg.smoke:
        tokenizer, _, student = load_smoke_models(cfg)
        # Load trained weights if available
        ckpt_path = model_dir / "checkpoint.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            student.load_state_dict(ckpt["student"])
    else:
        tokenizer = load_tokenizer(cfg.student.model_name)
        ckpt_path = model_dir / "checkpoint.pt"
        if ckpt_path.exists():
            student = AutoModelForCausalLM.from_pretrained(cfg.student.model_name)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            student.load_state_dict(ckpt["student"])
            log.info("Loaded trained checkpoint from %s", ckpt_path)
        elif model_dir.exists() and (model_dir / "config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            student = AutoModelForCausalLM.from_pretrained(model_dir)
        else:
            student = AutoModelForCausalLM.from_pretrained(cfg.student.model_name)
            log.warning("No trained model found at %s, using pretrained", model_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student.to(device)
    student.eval()

    results_dir = Path("results") / cfg.pair_name / experiment
    results_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = Path(cfg.data.processed_dir)
    max_new = cfg.eval.max_new_tokens or (cfg.eval.max_length - cfg.data.max_prompt_len)

    for test_set in cfg.eval.test_sets:
        test_file = processed_dir / f"{test_set}.jsonl"
        if not test_file.exists():
            log.warning("Test file not found: %s", test_file)
            continue

        test_data = _load_test_data(test_file)

        for seed in cfg.eval.seeds:
            set_seed(seed)
            out_file = results_dir / f"{test_set}_seed{seed}.jsonl"

            if out_file.exists():
                log.info("Eval output already exists: %s", out_file)
                continue

            log.info("Evaluating %s on %s (seed=%d)...", experiment, test_set, seed)
            results = []

            for item in tqdm(test_data, desc=f"{test_set} seed={seed}"):
                prompt = item["prompt"]
                reference = item["output"]

                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                if input_ids.shape[1] > cfg.data.max_prompt_len:
                    input_ids = input_ids[:, :cfg.data.max_prompt_len]

                with torch.no_grad():
                    if cfg.eval.do_sample:
                        gen = student.generate(
                            input_ids,
                            max_new_tokens=max_new,
                            do_sample=True,
                            temperature=cfg.eval.temperature,
                            top_p=cfg.eval.top_p,
                            top_k=cfg.eval.top_k,
                        )
                    else:
                        gen = student.generate(input_ids, max_new_tokens=max_new, do_sample=False)

                pred = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
                rouge = compute_rouge_l(pred, reference)

                results.append({
                    "id": item.get("id", ""),
                    "prompt": prompt,
                    "reference": reference,
                    "prediction": pred,
                    "rouge_l": rouge,
                })

            with out_file.open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            avg_rouge = sum(r["rouge_l"] for r in results) / max(len(results), 1)
            log.info("%s/%s seed=%d: ROUGE-L=%.2f", experiment, test_set, seed, avg_rouge)
