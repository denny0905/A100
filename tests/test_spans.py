"""Tests for span extraction per spec: 'The red car overtook the truck'."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from mta.data.spans import SpanExtractor


@pytest.fixture
def tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def extractor(tokenizer):
    return SpanExtractor(
        tokenizer=tokenizer,
        spacy_model="en_core_web_sm",
        max_vp_words=12,
        allow_overlap=False,
        max_spans=256,
    )


SENTENCE = "The red car overtook the truck"


class TestWordSpans:
    def test_word_count(self, extractor):
        """6 content words (no punctuation): The, red, car, overtook, the, truck."""
        spans = extractor.extract(SENTENCE)
        # spaCy should find 6 non-punct, non-space tokens
        # "The", "red", "car", "overtook", "the", "truck"
        assert len(spans["word"]) == 6, f"Expected 6 word spans, got {len(spans['word'])}: {spans['word']}"

    def test_word_spans_cover_content(self, extractor, tokenizer):
        """Every word span maps back to a valid substring of the text."""
        spans = extractor.extract(SENTENCE)
        encoding = tokenizer(SENTENCE, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding["offset_mapping"]

        for start, end in spans["word"]:
            # Token indices should be valid
            assert 0 <= start < end <= len(offsets)
            # Reconstruct text covered by span
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            covered = SENTENCE[char_start:char_end].strip()
            assert len(covered) > 0, f"Empty span at tokens [{start}:{end}]"


class TestPhraseSpans:
    def test_np_and_vp_present(self, extractor):
        """Should have NP 'The red car' and VP 'overtook the truck'."""
        spans = extractor.extract(SENTENCE)
        assert len(spans["phrase"]) >= 2, f"Expected ≥2 phrase spans, got {len(spans['phrase'])}"

    def test_phrases_disjoint(self, extractor):
        """Phrase spans should not overlap (allow_overlap=False)."""
        spans = extractor.extract(SENTENCE)
        phrase = sorted(spans["phrase"], key=lambda x: x[0])
        for i in range(len(phrase) - 1):
            assert phrase[i][1] <= phrase[i + 1][0], (
                f"Phrase spans overlap: {phrase[i]} and {phrase[i + 1]}"
            )

    def test_phrase_spans_map_to_correct_text(self, extractor, tokenizer):
        """NP should be 'The red car', VP should contain 'overtook'."""
        spans = extractor.extract(SENTENCE)
        encoding = tokenizer(SENTENCE, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding["offset_mapping"]

        texts = []
        for start, end in spans["phrase"]:
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            texts.append(SENTENCE[char_start:char_end].strip())

        text_str = " | ".join(texts)
        has_np = any("car" in t.lower() for t in texts)
        has_vp = any("overtook" in t.lower() for t in texts)
        assert has_np, f"Missing NP containing 'car' in: {text_str}"
        assert has_vp, f"Missing VP containing 'overtook' in: {text_str}"


class TestCharToTokenMapping:
    def test_roundtrip(self, extractor, tokenizer):
        """Every token in a span maps back to a valid character range."""
        spans = extractor.extract(SENTENCE)
        encoding = tokenizer(SENTENCE, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding["offset_mapping"]

        for span_type in ("word", "phrase"):
            for start, end in spans[span_type]:
                for tidx in range(start, end):
                    ts, te = offsets[tidx]
                    assert te > ts or (ts == 0 and te == 0), (
                        f"Token {tidx} has invalid offsets ({ts}, {te})"
                    )

    def test_bos_pad_not_in_spans(self, extractor, tokenizer):
        """BOS/special tokens with offset (0,0) should not appear in any span."""
        text = SENTENCE
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        offsets = encoding["offset_mapping"]

        special_indices = {i for i, (s, e) in enumerate(offsets) if s == 0 and e == 0}

        if not special_indices:
            pytest.skip("No special tokens with (0,0) offset for this tokenizer")

        spans = extractor.extract(text)
        for span_type in ("word", "phrase"):
            for start, end in spans[span_type]:
                for tidx in range(start, end):
                    assert tidx not in special_indices, (
                        f"Special token {tidx} found in {span_type} span"
                    )


class TestTensors:
    def test_to_tensors(self, extractor):
        spans = extractor.extract(SENTENCE)
        tensors = extractor.to_tensors(spans)

        assert tensors["word"].dtype == torch.long
        assert tensors["phrase"].dtype == torch.long
        assert tensors["word"].ndim == 2
        assert tensors["word"].shape[1] == 2

    def test_empty_spans(self, extractor):
        """Edge case: very short text might have no phrase spans."""
        spans = {"word": [], "phrase": []}
        tensors = extractor.to_tensors(spans)
        assert tensors["word"].shape == (0, 2)
        assert tensors["phrase"].shape == (0, 2)


class TestBatch:
    def test_batch_extraction(self, extractor):
        texts = [SENTENCE, "I like apples.", "Run fast."]
        results = extractor.extract_batch(texts)
        assert len(results) == 3
        for r in results:
            assert "word" in r
            assert "phrase" in r


class TestMaxLen:
    def test_spans_respect_max_len(self, extractor, tokenizer):
        long_text = SENTENCE + " " + SENTENCE + " " + SENTENCE
        max_len = 8
        spans = extractor.extract(long_text, max_len=max_len)
        for span_type in ("word", "phrase"):
            for start, end in spans[span_type]:
                assert end <= max_len, (
                    f"{span_type} span [{start},{end}) exceeds max_len={max_len}"
                )
