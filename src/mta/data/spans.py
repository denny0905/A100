"""Span extraction: word and phrase spans, char-to-token mapping, caching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class SpanExtractor:
    """Extract word and phrase spans from text, mapped to token indices."""

    def __init__(
        self,
        tokenizer: Any,
        spacy_model: str = "en_core_web_sm",
        max_vp_words: int = 12,
        allow_overlap: bool = False,
        max_spans: int = 256,
    ) -> None:
        import spacy

        self.tokenizer = tokenizer
        self.max_vp_words = max_vp_words
        self.allow_overlap = allow_overlap
        self.max_spans = max_spans
        self.nlp = spacy.load(
            spacy_model,
            disable=["ner", "lemmatizer"],
        )

    def _extract_word_spans_chars(self, doc: Any) -> list[tuple[int, int]]:
        """Extract word-level char spans, skipping punctuation and whitespace."""
        spans = []
        for tok in doc:
            if tok.is_punct or tok.is_space:
                continue
            spans.append((tok.idx, tok.idx + len(tok.text)))
        return spans

    def _extract_phrase_spans_chars(self, doc: Any) -> list[tuple[int, int]]:
        """Extract NP and VP char spans."""
        np_spans = [(chunk.start_char, chunk.end_char) for chunk in doc.noun_chunks]

        vp_spans = []
        for tok in doc:
            if tok.pos_ != "VERB":
                continue

            left_deps = {"neg", "aux", "auxpass"}
            right_deps = {"dobj", "obj", "attr", "prt", "dative", "acomp", "oprd"}

            start_tok = tok
            for child in tok.children:
                if child.dep_ in left_deps and child.i < tok.i:
                    if child.i == start_tok.i - 1 or child.i < start_tok.i:
                        start_tok = min(start_tok, child, key=lambda t: t.i)

            end_tok = tok
            for child in tok.children:
                if child.dep_ in right_deps and child.i > tok.i:
                    subtree = list(child.subtree)
                    rightmost = max(subtree, key=lambda t: t.i)
                    if rightmost.i > end_tok.i:
                        end_tok = rightmost

            vp_tokens = [t for t in doc if start_tok.i <= t.i <= end_tok.i]
            if len(vp_tokens) > self.max_vp_words:
                prt_children = [c for c in tok.children if c.dep_ == "prt"]
                if prt_children:
                    end_tok = max(prt_children, key=lambda t: t.i)
                    vp_tokens = [t for t in doc if tok.i <= t.i <= end_tok.i]
                else:
                    vp_tokens = [tok]
                    end_tok = tok

            start_char = vp_tokens[0].idx
            last = vp_tokens[-1]
            end_char = last.idx + len(last.text)
            vp_spans.append((start_char, end_char))

        if not self.allow_overlap:
            np_spans, vp_spans = self._remove_overlaps(np_spans, vp_spans)

        return np_spans + vp_spans

    @staticmethod
    def _remove_overlaps(
        np_spans: list[tuple[int, int]],
        vp_spans: list[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Remove NPs fully contained within any VP."""
        filtered_nps = []
        for ns, ne in np_spans:
            contained = any(vs <= ns and ne <= ve for vs, ve in vp_spans)
            if not contained:
                filtered_nps.append((ns, ne))
        return filtered_nps, vp_spans

    def _map_char_to_token(
        self,
        text: str,
        char_spans: list[tuple[int, int]],
        max_len: int | None = None,
    ) -> list[tuple[int, int]]:
        """Map character spans to token index spans using offset_mapping."""
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len or 99999,
        )
        offsets = encoding["offset_mapping"]

        token_spans = []
        for cs, ce in char_spans:
            tok_start = None
            tok_end = None

            for tidx, (ts, te) in enumerate(offsets):
                if ts == 0 and te == 0:
                    continue

                # Strip leading whitespace from token offset for BPE alignment
                effective_ts = ts
                while effective_ts < te and text[effective_ts] == " ":
                    effective_ts += 1

                # Token belongs to span if its effective char range intersects
                if effective_ts < ce and te > cs:
                    if tok_start is None:
                        tok_start = tidx
                    tok_end = tidx + 1

            if tok_start is not None and tok_end is not None:
                if max_len is None or tok_end <= max_len:
                    token_spans.append((tok_start, tok_end))

        return token_spans

    def extract(
        self,
        text: str,
        max_len: int | None = None,
    ) -> dict[str, list[tuple[int, int]]]:
        """Extract word and phrase spans for a single text."""
        doc = self.nlp(text)

        word_chars = self._extract_word_spans_chars(doc)
        phrase_chars = self._extract_phrase_spans_chars(doc)

        word_tokens = self._map_char_to_token(text, word_chars, max_len)
        phrase_tokens = self._map_char_to_token(text, phrase_chars, max_len)

        if len(word_tokens) > self.max_spans:
            log.warning(
                "Truncating word spans from %d to %d", len(word_tokens), self.max_spans
            )
            word_tokens = word_tokens[: self.max_spans]
        if len(phrase_tokens) > self.max_spans:
            log.warning(
                "Truncating phrase spans from %d to %d",
                len(phrase_tokens),
                self.max_spans,
            )
            phrase_tokens = phrase_tokens[: self.max_spans]

        return {"word": word_tokens, "phrase": phrase_tokens}

    def extract_batch(
        self,
        texts: list[str],
        max_len: int | None = None,
    ) -> list[dict[str, list[tuple[int, int]]]]:
        """Extract spans for a batch of texts using spaCy pipe."""
        docs = list(self.nlp.pipe(texts))
        results = []
        for doc, text in zip(docs, texts):
            word_chars = self._extract_word_spans_chars(doc)
            phrase_chars = self._extract_phrase_spans_chars(doc)
            word_tokens = self._map_char_to_token(text, word_chars, max_len)
            phrase_tokens = self._map_char_to_token(text, phrase_chars, max_len)

            if len(word_tokens) > self.max_spans:
                word_tokens = word_tokens[: self.max_spans]
            if len(phrase_tokens) > self.max_spans:
                phrase_tokens = phrase_tokens[: self.max_spans]

            results.append({"word": word_tokens, "phrase": phrase_tokens})
        return results

    def to_tensors(
        self, spans: dict[str, list[tuple[int, int]]]
    ) -> dict[str, torch.LongTensor]:
        """Convert span lists to tensors of shape [K, 2]."""
        result = {}
        for key in ("word", "phrase"):
            s = spans.get(key, [])
            if s:
                result[key] = torch.tensor(s, dtype=torch.long)
            else:
                result[key] = torch.zeros(0, 2, dtype=torch.long)
        return result


def _cache_path(cache_dir: str, tokenizer_name: str, split: str) -> Path:
    safe_name = tokenizer_name.replace("/", "__")
    return Path(cache_dir) / safe_name / f"{split}.pt"


def precompute_all(cfg: DictConfig) -> None:
    """Precompute and cache spans for all processed splits."""
    from transformers import AutoTokenizer

    processed_dir = Path(cfg.data.processed_dir)
    tokenizer_name = cfg.student.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    extractor = SpanExtractor(
        tokenizer=tokenizer,
        spacy_model=cfg.spans.spacy_model,
        max_vp_words=cfg.spans.max_vp_words,
        allow_overlap=cfg.spans.allow_overlap,
        max_spans=cfg.spans.max_spans,
    )

    splits = [f.stem for f in processed_dir.glob("*.jsonl")]
    if not splits:
        log.warning("No JSONL files found in %s", processed_dir)
        return

    for split in splits:
        cache = _cache_path(cfg.spans.cache_dir, tokenizer_name, split)
        if cache.exists():
            log.info("Span cache exists for %s, skipping.", split)
            continue

        from mta.utils.io import read_jsonl

        data = read_jsonl(processed_dir / f"{split}.jsonl")
        texts = [item["prompt"] + item["output"] for item in data]
        max_len = cfg.data.max_len

        log.info("Extracting spans for %s (%d samples)...", split, len(texts))
        all_spans = extractor.extract_batch(texts, max_len=max_len)

        tensor_spans = [extractor.to_tensors(s) for s in all_spans]

        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor_spans, cache)
        log.info("Cached spans for %s -> %s", split, cache)


def load_cached_spans(
    cache_dir: str, tokenizer_name: str, split: str
) -> list[dict[str, torch.LongTensor]]:
    """Load precomputed spans from cache."""
    cache = _cache_path(cache_dir, tokenizer_name, split)
    if not cache.exists():
        raise FileNotFoundError(f"Span cache not found: {cache}")
    return torch.load(cache, weights_only=False)
