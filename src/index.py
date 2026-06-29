from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import faiss
import numpy as np

from .arabic_text import content_fingerprint
from .config import Config

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2


@dataclass
class SegmentMeta:
    id: int
    audio_file: str
    start: float
    end: float
    text: str
    fingerprint: str = ""


@dataclass
class IndexManifest:
    schema_version: int
    embedder_model_id: str
    dim: int
    next_id: int
    count: int = 0
    extras: dict = field(default_factory=dict)


class FaissSegmentIndex:
    """FAISS index plus metadata for Arabic audio segments."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        idx_cfg = cfg.raw["index"]
        self.dir = Path(idx_cfg["dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.faiss_path = self.dir / idx_cfg["faiss_file"]
        self.meta_path = self.dir / idx_cfg["meta_file"]
        self.manifest_path = self.dir / "manifest.json"
        self.dim = idx_cfg["dim"]
        self.embedder_model_id: str = cfg.raw["embed"]["model_id"]

        self.index: faiss.Index
        self.meta_by_id: dict[int, SegmentMeta] = {}
        self.fingerprints: dict[str, int] = {}
        self.next_id: int = 0
        self._load()

    def _new_index(self) -> faiss.Index:
        return faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))

    def _fresh(self) -> None:
        self.index = self._new_index()
        self.meta_by_id = {}
        self.fingerprints = {}
        self.next_id = 0

    def _load(self) -> None:
        if not self.manifest_path.exists():
            # Legacy or empty — refuse to read an unversioned FAISS file.
            if self.faiss_path.exists():
                log.warning(
                    "Found legacy FAISS index without manifest at %s. "
                    "Ignoring and starting fresh — re-ingest your audio.",
                    self.faiss_path,
                )
            self._fresh()
            return

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            log.warning(
                "Index schema_version=%s doesn't match current=%s; rebuilding.",
                manifest.get("schema_version"), SCHEMA_VERSION,
            )
            self._fresh()
            return
        if manifest.get("embedder_model_id") != self.embedder_model_id:
            raise RuntimeError(
                f"Index was built with embedder '{manifest.get('embedder_model_id')}' "
                f"but config uses '{self.embedder_model_id}'. Delete "
                f"'{self.dir}' to rebuild, or revert the embedder."
            )
        if manifest.get("dim") != self.dim:
            raise RuntimeError(
                f"Index dim {manifest.get('dim')} != config dim {self.dim}. "
                f"Delete '{self.dir}' to rebuild."
            )

        if self.faiss_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
        else:
            self.index = self._new_index()

        self.meta_by_id = {}
        self.fingerprints = {}
        if self.meta_path.exists():
            for line in self.meta_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                m = SegmentMeta(
                    id=int(d["id"]),
                    audio_file=d["audio_file"],
                    start=float(d["start"]),
                    end=float(d["end"]),
                    text=d["text"],
                    fingerprint=d.get("fingerprint", ""),
                )
                self.meta_by_id[m.id] = m
                if m.fingerprint:
                    self.fingerprints[m.fingerprint] = m.id

        self.next_id = int(manifest.get("next_id", max(self.meta_by_id, default=-1) + 1))

    def save(self) -> None:
        """Atomically persist FAISS index, metadata, and manifest."""
        faiss.write_index(self.index, str(self.faiss_path))
        tmp_meta = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
        with tmp_meta.open("w", encoding="utf-8") as f:
            for m in self.meta_by_id.values():
                f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
        tmp_meta.replace(self.meta_path)

        manifest = IndexManifest(
            schema_version=SCHEMA_VERSION,
            embedder_model_id=self.embedder_model_id,
            dim=self.dim,
            next_id=self.next_id,
            count=len(self.meta_by_id),
        )
        self.manifest_path.write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(
        self,
        vectors: np.ndarray,
        segments: list[dict],
        audio_file: str,
    ) -> int:
        """Add passage vectors and metadata, skipping duplicates by fingerprint."""
        assert vectors.shape[0] == len(segments), (
            f"vectors/segments count mismatch: {vectors.shape[0]} vs {len(segments)}"
        )
        if vectors.shape[0] == 0:
            return 0

        new_vecs: list[np.ndarray] = []
        new_ids: list[int] = []
        for vec, s in zip(vectors, segments):
            text = s["text"]
            fp = content_fingerprint(text, audio_file, s["start"])
            if fp in self.fingerprints:
                continue
            mid = self.next_id
            self.next_id += 1
            self.meta_by_id[mid] = SegmentMeta(
                id=mid,
                audio_file=audio_file,
                start=float(s["start"]),
                end=float(s["end"]),
                text=text,
                fingerprint=fp,
            )
            self.fingerprints[fp] = mid
            new_vecs.append(vec.astype(np.float32))
            new_ids.append(mid)

        if not new_ids:
            return 0

        vecs_arr = np.vstack(new_vecs).astype(np.float32)
        ids_arr = np.array(new_ids, dtype=np.int64)
        self.index.add_with_ids(vecs_arr, ids_arr)
        return len(new_ids)

    def remove_by_audio(self, audio_file: str) -> int:
        """Remove all segments for a given audio file and return how many were removed."""
        ids = [m.id for m in self.meta_by_id.values() if m.audio_file == audio_file]
        if not ids:
            return 0
        id_arr = np.array(ids, dtype=np.int64)
        self.index.remove_ids(faiss.IDSelectorBatch(id_arr))
        for mid in ids:
            m = self.meta_by_id.pop(mid, None)
            if m and m.fingerprint in self.fingerprints:
                self.fingerprints.pop(m.fingerprint, None)
        return len(ids)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def search(
        self, query_vec: np.ndarray, k: int = 5
    ) -> list[tuple[float, SegmentMeta]]:
        """Search by embedding and return top-k matches."""
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        q = query_vec.reshape(1, -1).astype(np.float32)
        D, I = self.index.search(q, k)
        out: list[tuple[float, SegmentMeta]] = []
        for j in range(len(I[0])):
            mid = int(I[0][j])
            if mid < 0:
                continue
            m = self.meta_by_id.get(mid)
            if m is None:
                continue
            out.append((float(D[0][j]), m))
        return out

    def iter_meta(self):
        """Iterate over all SegmentMeta in id order."""
        for mid in sorted(self.meta_by_id):
            yield self.meta_by_id[mid]
