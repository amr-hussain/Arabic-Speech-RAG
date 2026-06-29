from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from .arabic_text import tokenize
from .config import Config


class BM25Index:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        idx_cfg = cfg.raw["index"]
        self.dir = Path(idx_cfg["dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / idx_cfg.get("bm25_file", "bm25.jsonl")

        self.ids: list[int] = []
        self.corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            self.ids.append(int(d["id"]))
            self.corpus.append(list(d["tokens"]))
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)

    def save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for mid, toks in zip(self.ids, self.corpus):
                f.write(json.dumps({"id": mid, "tokens": toks}, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def rebuild_from(self, segments_iter) -> None:
        """Rebuild the BM25 corpus from an iterable of SegmentMeta."""
        self.ids = []
        self.corpus = []
        for m in segments_iter:
            toks = tokenize(m.text)
            if not toks:
                continue
            self.ids.append(int(m.id))
            self.corpus.append(toks)
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def search(self, query: str, k: int = 50) -> list[tuple[float, int]]:
        """Return up to k (score, segment_id) pairs ranked by BM25."""
        if not self.bm25 or not self.corpus:
            return []
        q_toks = tokenize(query)
        if not q_toks:
            return []
        scores = self.bm25.get_scores(q_toks)
        if scores.size == 0:
            return []

        # Top-k by score — argsort descending.
        k = min(k, len(scores))
        # np argpartition is faster but argsort is fine for our scale.
        import numpy as np
        idx = np.argsort(-scores)[:k]
        return [
            (float(scores[i]), int(self.ids[i]))
            for i in idx
            if scores[i] > 0.0
        ]
