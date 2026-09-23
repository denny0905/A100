"""Tests for data pipeline: dataset, collator, prompt template."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from mta.data.dataset import InstructionCollator, InstructionDataset
from mta.data.prepare import format_prompt


@pytest.fixture
def tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def sample_data(tmp_path: Path) -> Path:
    data = [
        {
            "id": "0",
            "prompt": format_prompt("Say hello.", ""),
            "output": "Hello there!",
        },
        {
            "id": "1",
            "prompt": format_prompt("Translate.", "Good morning"),
            "output": "Bonjour",
        },
        {
            "id": "2",
            "prompt": format_prompt("Summarize.", "A long text here."),
            "output": "Short summary.",
        },
    ]
    path = tmp_path / "test.jsonl"
    with path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path


class TestPromptTemplate:
    def test_no_input(self):
        p = format_prompt("Say hello.")
        assert "### Instruction:\nSay hello." in p
        assert "### Input:" not in p
        assert p.endswith("### Response:\n")

    def test_with_input(self):
        p = format_prompt("Translate.", "Hello")
        assert "### Input:\nHello" in p
        assert "### Instruction:\nTranslate." in p

    def test_empty_input_same_as_no_input(self):
        assert format_prompt("X", "") == format_prompt("X")
        assert format_prompt("X", "   ") == format_prompt("X")


class TestDataset:
    def test_loads_and_tokenizes(self, tokenizer, sample_data):
        ds = InstructionDataset(sample_data, tokenizer, max_len=512, max_prompt_len=256)
        assert len(ds) == 3
        item = ds[0]
        assert "input_ids" in item
        assert "prompt_length" in item
        assert item["input_ids"].dtype == torch.long

    def test_truncation(self, tokenizer, sample_data):
        ds = InstructionDataset(sample_data, tokenizer, max_len=64, max_prompt_len=32)
        item = ds[0]
        assert len(item["input_ids"]) <= 64


class TestCollator:
    def test_batch_shapes(self, tokenizer, sample_data):
        ds = InstructionDataset(sample_data, tokenizer, max_len=128, max_prompt_len=64)
        collator = InstructionCollator(tokenizer, max_len=128)
        batch = collator([ds[i] for i in range(3)])

        assert batch["input_ids"].shape[0] == 3
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        assert batch["labels"].shape == batch["input_ids"].shape
        assert batch["response_mask"].shape == batch["input_ids"].shape

    def test_labels_mask_prompt_and_padding(self, tokenizer, sample_data):
        ds = InstructionDataset(sample_data, tokenizer, max_len=128, max_prompt_len=64)
        collator = InstructionCollator(tokenizer, max_len=128)
        batch = collator([ds[i] for i in range(3)])

        for i in range(3):
            plen = batch["prompt_lengths"][i].item()
            labels = batch["labels"][i]
            attn = batch["attention_mask"][i]

            # Prompt tokens should be masked (-100)
            assert (labels[:plen] == -100).all(), "Prompt should be masked in labels"

            # Padding should be masked (-100)
            pad_start = attn.sum().item()
            if pad_start < len(labels):
                assert (labels[pad_start:] == -100).all(), "Padding should be masked"

            # Response should have valid labels
            resp_mask = batch["response_mask"][i]
            resp_labels = labels[resp_mask.bool()]
            assert (resp_labels != -100).all(), "Response labels should not be masked"

    def test_response_mask_only_response(self, tokenizer, sample_data):
        ds = InstructionDataset(sample_data, tokenizer, max_len=128, max_prompt_len=64)
        collator = InstructionCollator(tokenizer, max_len=128)
        batch = collator([ds[0]])

        plen = batch["prompt_lengths"][0].item()
        resp = batch["response_mask"][0]
        seq_len = batch["attention_mask"][0].sum().item()

        assert resp[:plen].sum() == 0, "No response flag in prompt region"
        assert resp[seq_len:].sum() == 0, "No response flag in padding region"
        assert resp[plen:seq_len].sum() > 0, "Response region should be flagged"

    def test_pad_token_set(self, tokenizer, sample_data):
        ds = InstructionDataset(sample_data, tokenizer, max_len=128, max_prompt_len=64)
        collator = InstructionCollator(tokenizer, max_len=128)
        batch = collator([ds[0], ds[1]])
        # Should not crash — collator sets pad_token if missing
        assert batch["input_ids"].shape[0] == 2
