from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict

import torch
from huggingface_hub import InferenceClient
from transformers import pipeline

from .arabic_text import (
    collapse_repeats,
    looks_like_hallucination,
    normalize_arabic,
    similarity_ratio,
)
from .audio import SR, chunk, load_audio
from .config import Config


class Segment(TypedDict):
    start: float
    end: float
    text: str


_LONGFORM_ONLY_KEYS = {
    "condition_on_prev_tokens",
    "compression_ratio_threshold",
    "logprob_threshold",
    "no_speech_threshold",
}


def _build_generate_kwargs(asr_cfg: dict, long_form: bool) -> dict:
    """Build generate kwargs for Whisper based on config and decoding mode."""
    gk: dict = {
        "language": asr_cfg.get("language", "ar"),
        "task": "transcribe",
    }
    extra = asr_cfg.get("generation_kwargs") or {}
    for k, v in extra.items():
        if v is None:
            continue
        if not long_form and k in _LONGFORM_ONLY_KEYS:
            continue
        if k == "temperature":
            if isinstance(v, list):
                v = tuple(v)
            # Scalarise for short-form — tuple is long-form-only and will raise.
            if not long_form and isinstance(v, tuple):
                v = float(v[0]) if v else 0.0
        gk[k] = v
    return gk


class ASR(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> list[Segment]:
        ...


class LocalWhisperASR(ASR):
    """ASR backend that runs Whisper locally."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        asr_cfg = cfg.raw["asr"]

        # Pipeline-level chunking forces short-form decoding per chunk and
        # disables hallucination thresholds + temperature fallback. Setting
        # chunk_length_s to 0 / null opts into Whisper's native long-form
        # algorithm, which is the recommended path for lectures & calls.
        chunk_s = asr_cfg.get("chunk_length_s") or 0
        stride_s = asr_cfg.get("stride_length_s") or 0
        long_form = chunk_s <= 0

        pipe_kwargs: dict = {
            "task": "automatic-speech-recognition",
            "model": asr_cfg["model_id"],
            "torch_dtype": cfg.torch_dtype,
            "device": 0 if cfg.device == "cuda" else -1,
            "batch_size": asr_cfg.get("batch_size", 1),
            "return_timestamps": True,
            "generate_kwargs": _build_generate_kwargs(asr_cfg, long_form=long_form),
        }
        if not long_form:
            pipe_kwargs["chunk_length_s"] = chunk_s
            if stride_s > 0:
                pipe_kwargs["stride_length_s"] = stride_s

        self.pipe = pipeline(**pipe_kwargs)
        self.long_form = long_form

    def transcribe(self, audio_path: str | Path) -> list[Segment]:
        result = self.pipe(str(audio_path))
        segs: list[Segment] = []
        for c in result.get("chunks", []):
            ts = c.get("timestamp", (None, None))
            start = float(ts[0]) if ts[0] is not None else 0.0
            end = float(ts[1]) if ts[1] is not None else start
            text = (c.get("text") or "").strip()
            if text:
                segs.append({"start": start, "end": end, "text": text})
        if not segs:
            text = (result.get("text") or "").strip()
            if text:
                segs.append({"start": 0.0, "end": 0.0, "text": text})
        return segs


class HFInferenceASR(ASR):
    """ASR backend that sends audio chunks to a Hugging Face Inference endpoint."""

    def __init__(self, cfg: Config):
        token = cfg.hf_token()
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not set. Add it to your .env file to use hf_api backend."
            )
        asr_cfg = cfg.raw["asr"]
        provider = asr_cfg.get("hf_provider", "fal-ai")

        # InferenceClient gained `provider=` / `api_key=` kwargs in
        # huggingface_hub 0.29+. Older versions silently drop them and the
        # legacy Serverless endpoint 404s for Whisper. Fail loudly so the
        # user upgrades rather than debugging a mysterious 404.
        import inspect
        params = inspect.signature(InferenceClient.__init__).parameters
        if provider and "provider" not in params:
            raise RuntimeError(
                "Your installed `huggingface_hub` is too old for the Inference "
                "Providers API (needed to reach whisper-large-v3 via fal-ai). "
                "Upgrade with:  pip install -U 'huggingface_hub>=0.29.0'"
            )

        # `provider=None` routes through HF's own inference; any other value
        # (e.g. "fal-ai", "replicate", "together") routes to that provider.
        if provider:
            self.client = InferenceClient(provider=provider, api_key=token)
        else:
            self.client = InferenceClient(api_key=token)
        self.model = asr_cfg["hf_api_model"]
        # HF API has a per-request size limit, so we always chunk.
        # chunk_length_s=0 in config (for local long-form) falls back to 30s.
        cs = asr_cfg.get("chunk_length_s") or 0
        self.chunk_s = float(cs) if cs > 0 else 30.0

    def transcribe(self, audio_path: str | Path) -> list[Segment]:
        import soundfile as sf

        y = load_audio(audio_path)
        segs: list[Segment] = []
        for ch in chunk(y, chunk_s=self.chunk_s, overlap_s=0.0):
            buf = io.BytesIO()
            sf.write(buf, ch.samples, SR, format="WAV")
            buf.seek(0)
            out = self.client.automatic_speech_recognition(
                buf.read(), model=self.model
            )
            text = out.text if hasattr(out, "text") else out["text"]
            text = (text or "").strip()
            if text:
                segs.append({"start": ch.start_s, "end": ch.end_s, "text": text})
        return segs


def build_asr(cfg: Config) -> ASR:
    backend = cfg.raw["asr"]["backend"]
    if backend == "local":
        return LocalWhisperASR(cfg)
    if backend == "hf_api":
        return HFInferenceASR(cfg)
    raise ValueError(f"Unknown asr.backend: '{backend}'. Must be 'local' or 'hf_api'.")


# ── Post-ASR cleanup & passage merger ───────────────────────────────────────

def cleanup_segments(
    segs: list[Segment],
    *,
    max_run: int = 2,
    hallucination_ratio: float = 0.6,
    adjacent_dup_ratio: float = 0.85,
    min_chars: int = 2,
) -> list[Segment]:
    """Clean raw Whisper segments using basic repetition and hallucination filters."""
    out: list[Segment] = []
    for s in segs:
        raw = (s["text"] or "").strip()
        if not raw:
            continue
        # Detect hallucination on raw text (before repeats are collapsed),
        # otherwise "شكرا شكرا شكرا شكرا شكرا شكرا" would get trimmed to
        # "شكرا شكرا" and slip past the dominance heuristic.
        if looks_like_hallucination(raw, max_ratio=hallucination_ratio):
            continue
        text = collapse_repeats(raw, max_run=max_run).strip()
        if len(text) < min_chars:
            continue
        if out and similarity_ratio(out[-1]["text"], text) >= adjacent_dup_ratio:
            # Extend timestamp to cover dup region, drop the text
            out[-1]["end"] = max(out[-1]["end"], float(s["end"]))
            continue
        out.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": text,
        })
    return out


def merge_passages(
    segs: list[Segment],
    *,
    target_words: int = 60,
    max_words: int = 100,
    max_seconds: float = 30.0,
    overlap_segments: int = 1,
) -> list[Segment]:
    """Merge cleaned segments into longer passages for retrieval."""
    if not segs:
        return []
    if target_words <= 0:
        target_words = 1
    overlap_segments = max(0, int(overlap_segments))

    passages: list[Segment] = []
    i = 0
    n = len(segs)

    while i < n:
        cur: list[Segment] = []
        cur_words = 0
        first_start = segs[i]["start"]
        j = i
        while j < n:
            s = segs[j]
            w = len(s["text"].split())
            span = float(s["end"]) - first_start
            # Close conditions — but always include at least one segment
            if cur and (
                cur_words + w > max_words
                or span > max_seconds
                or cur_words >= target_words
            ):
                break
            cur.append(s)
            cur_words += w
            j += 1

        if not cur:
            # Safety net — shouldn't happen because we always take one.
            i = j + 1
            continue

        text = " ".join(c["text"] for c in cur).strip()
        passages.append({
            "start": float(cur[0]["start"]),
            "end": float(cur[-1]["end"]),
            "text": text,
        })

        # Step forward, leaving `overlap_segments` of tail for the next pass.
        consumed = len(cur)
        step = max(1, consumed - overlap_segments)
        i += step

    # Final safety pass: drop empty, collapse repeats inside merged passages.
    cleaned: list[Segment] = []
    for p in passages:
        t = collapse_repeats(p["text"], max_run=2).strip()
        if not t:
            continue
        p["text"] = t
        cleaned.append(p)
    return cleaned
