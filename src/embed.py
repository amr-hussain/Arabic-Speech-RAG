from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import Config


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over token embeddings, ignoring padding."""
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class Embedder:
    """
    Multilingual E5 embedder for Arabic semantic search.

    IMPORTANT: E5 requires mandatory prefixes:
      - Passages (indexed segments): "passage: <text>"
      - Queries (search input):       "query: <text>"
    Omitting these significantly degrades retrieval quality.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        e = cfg.raw["embed"]
        self.tok = AutoTokenizer.from_pretrained(e["model_id"])
        self.model = AutoModel.from_pretrained(
            e["model_id"], torch_dtype=cfg.torch_dtype
        ).to(cfg.device).eval()
        self.max_len = e["max_length"]
        self.bs = e["batch_size"]
        self.qp = e["query_prefix"]
        self.pp = e["passage_prefix"]
        self.dim = cfg.raw["index"]["dim"]

    @torch.inference_mode()
    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        batches: list[np.ndarray] = []
        for i in range(0, len(texts), self.bs):
            batch = texts[i : i + self.bs]
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.cfg.device)
            h = self.model(**enc).last_hidden_state
            v = _mean_pool(h, enc["attention_mask"])
            v = torch.nn.functional.normalize(v, p=2, dim=1)
            batches.append(v.float().cpu().numpy())
        return np.concatenate(batches, axis=0)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Encode text segments for indexing. Adds mandatory 'passage: ' prefix."""
        return self._encode([self.pp + t for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a search query. Adds mandatory 'query: ' prefix. Returns shape (dim,)."""
        return self._encode([self.qp + text])[0]
