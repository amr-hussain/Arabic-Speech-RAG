from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    raw: dict
    device: str          # "cuda" or "cpu"
    torch_dtype: torch.dtype

    @classmethod
    def load(cls, path: str | Path = "configs/config.yaml") -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        dev = raw.get("device", "auto")
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = (
            torch.float16
            if (dev == "cuda" and raw.get("dtype") == "fp16")
            else torch.float32
        )
        return cls(raw=raw, device=dev, torch_dtype=dtype)

    def hf_token(self) -> str | None:
        return os.getenv("HF_TOKEN")
