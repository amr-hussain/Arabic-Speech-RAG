from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import Config


class Summarizer:
    """
    Arabic-capable hierarchical summarizer backed by mT5-XLSum.

    For transcripts longer than `window_chars`, the text is split into
    windows, each window is summarized individually, then the partial
    summaries are combined and summarized again (two-level hierarchy).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        mid = cfg.raw["summarize"]["model_id"]
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            mid, torch_dtype=cfg.torch_dtype
        ).to(cfg.device).eval()

    @torch.inference_mode()
    def _summarize_once(self, text: str) -> str:
        s = self.cfg.raw["summarize"]
        inputs = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=s["max_input_tokens"],
        ).to(self.cfg.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=s["max_new_tokens"],
            min_new_tokens=int(s.get("min_new_tokens", 16)),
            num_beams=s["num_beams"],
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
        )
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def _split_into_windows(self, text: str, window: int) -> list[str]:
        """Split text into character-bounded windows on word boundaries."""
        words = text.split()
        parts: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for word in words:
            if cur_len + len(word) + 1 > window:
                parts.append(" ".join(cur))
                cur = [word]
                cur_len = len(word)
            else:
                cur.append(word)
                cur_len += len(word) + 1
        if cur:
            parts.append(" ".join(cur))
        return parts

    def summarize(self, text: str) -> str:
        """Summarize Arabic text, handling long inputs hierarchically."""
        if not text.strip():
            return ""
        window = self.cfg.raw["summarize"]["window_chars"]
        if len(text) <= window:
            return self._summarize_once(text)
        # Level 1: one short paragraph per chunk
        parts = self._split_into_windows(text, window)
        partials = [self._summarize_once(p) for p in parts]
        joined = "\n\n".join(partials)
        # Level 2: only collapse if the joined partials are still too long for
        # a single model pass; otherwise preserve the multi-paragraph output so
        # long transcripts get realistic multi-paragraph summaries.
        if len(joined) > window:
            return self._summarize_once(joined[:window])
        return joined
