from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16_000  # target sample rate


@dataclass
class AudioChunk:
    start_s: float
    end_s: float
    samples: np.ndarray  # float32, mono, 16 kHz


def load_audio(path: str | Path) -> np.ndarray:
    """Load any audio file and return mono float32 array @ 16 kHz."""
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y.astype(np.float32)


def chunk(
    y: np.ndarray,
    chunk_s: float = 30.0,
    overlap_s: float = 1.0,
) -> list[AudioChunk]:
    """Split audio into overlapping chunks with timestamps."""
    step = int((chunk_s - overlap_s) * SR)
    size = int(chunk_s * SR)
    out: list[AudioChunk] = []
    for start in range(0, max(1, len(y)), step):
        end = min(start + size, len(y))
        out.append(AudioChunk(start / SR, end / SR, y[start:end]))
        if end == len(y):
            break
    return out


def slice_segment(
    path: str | Path,
    start_s: float,
    end_s: float,
    out_path: str | Path,
) -> str:
    """Extract a time segment from an audio file and write to out_path."""
    y = load_audio(path)
    a = int(start_s * SR)
    b = int(end_s * SR)
    sf.write(str(out_path), y[a:b], SR)
    return str(out_path)
