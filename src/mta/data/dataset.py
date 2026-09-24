"""Dataset and collator for instruction-tuning / distillation."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    """JSONL dataset with prompt/output fields. Tokenizes once and caches to disk."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_len: int = 512,
        max_prompt_len: int = 256,
    ) -> None:
        self.max_len = max_len
        self.max_prompt_len = max_prompt_len
        path = Path(path)

        cache_path = self._cache_path(path, tokenizer, max_len, max_prompt_len)
        if cache_path.exists():
            log.info("Loading cached tokenized data from %s", cache_path)
            cached = torch.load(cache_path, weights_only=False)
            self.tokens = cached["tokens"]
            self.full_texts = cached["full_texts"]
        else:
            log.info("Pre-tokenizing %s ...", path)
            raw = self._load(path)
            self.tokens, self.full_texts = self._tokenize_all(raw, tokenizer)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"tokens": self.tokens, "full_texts": self.full_texts}, cache_path)
            log.info("Cached %d samples to %s", len(self.tokens), cache_path)

    @staticmethod
    def _cache_path(path: Path, tokenizer: Any, max_len: int, max_prompt_len: int) -> Path:
        key = f"{path.name}_{getattr(tokenizer, 'name_or_path', 'unk')}_{max_len}_{max_prompt_len}"
        h = hashlib.md5(key.encode()).hexdigest()[:8]
        return path.parent / f".cache_{path.stem}_{h}.pt"

    def _load(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _tokenize_all(self, data: list[dict[str, str]], tokenizer: Any) -> tuple[list[dict], list[str]]:
        tokens = []
        full_texts = []
        for item in data:
            prompt = item["prompt"]
            output = item["output"]

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            output_ids = tokenizer.encode(output, add_special_tokens=False)

            if len(prompt_ids) > self.max_prompt_len:
                prompt_ids = prompt_ids[: self.max_prompt_len]

            max_output_len = self.max_len - len(prompt_ids)
            if max_output_len <= 0:
                max_output_len = 1
            if len(output_ids) > max_output_len:
                output_ids = output_ids[:max_output_len]

            input_ids = prompt_ids + output_ids
            tokens.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "prompt_length": len(prompt_ids),
            })
            full_texts.append(prompt + output)
        return tokens, full_texts

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t = self.tokens[idx]
        return {
            "input_ids": t["input_ids"],
            "prompt_length": t["prompt_length"],
            "full_text": self.full_texts[idx],
        }


@dataclass
class InstructionCollator:
    """Pad-right collator that masks prompt and padding in labels."""

    tokenizer: Any
    max_len: int = 512

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids_list = [item["input_ids"] for item in batch]
        prompt_lengths = [item["prompt_length"] for item in batch]
        full_texts = [item["full_text"] for item in batch]

        max_seq_len = min(max(len(ids) for ids in input_ids_list), self.max_len)

        padded_input_ids = []
        attention_masks = []
        labels_list = []
        response_masks = []

        pad_id = self.tokenizer.pad_token_id

        for ids, plen in zip(input_ids_list, prompt_lengths):
            seq_len = len(ids)
            pad_len = max_seq_len - seq_len

            padded = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            attn = torch.cat([
                torch.ones(seq_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ])

            lab = padded.clone()
            lab[:plen] = -100
            lab[seq_len:] = -100

            resp = torch.zeros(max_seq_len, dtype=torch.long)
            resp[plen:seq_len] = 1

            padded_input_ids.append(padded)
            attention_masks.append(attn)
            labels_list.append(lab)
            response_masks.append(resp)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels_list),
            "response_mask": torch.stack(response_masks),
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "full_texts": full_texts,
        }
