from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Literal

from .asr import build_asr, cleanup_segments, merge_passages
from .bm25 import BM25Index
from .config import Config
from .embed import Embedder
from .index import FaissSegmentIndex, SegmentMeta
from .summarize import Summarizer

log = logging.getLogger(__name__)

SearchMode = Literal["semantic", "lexical", "hybrid"]


def _free_cuda() -> None:
    """Best-effort release of CUDA memory after a model is deleted."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


class Pipeline:
    """High-level Arabic audio pipeline for ingesting audio and running search."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.low_vram: bool = bool(cfg.raw.get("low_vram", False))
        self.idx = FaissSegmentIndex(cfg)
        self.bm25 = BM25Index(cfg)

        self._asr = None
        self._summ = None
        self._emb = None

        Path(cfg.raw["paths"]["transcripts_dir"]).mkdir(parents=True, exist_ok=True)
        Path(cfg.raw["paths"]["summaries_dir"]).mkdir(parents=True, exist_ok=True)

        if not self.low_vram:
            # Eager load (keeps all three resident for fastest repeated use)
            self._asr = build_asr(cfg)
            self._summ = Summarizer(cfg)
            self._emb = Embedder(cfg)

        # If FAISS is populated but BM25 is missing/empty, rebuild BM25 lazily
        # so hybrid search works after a crash or manual BM25 file deletion.
        if self.idx.ntotal > 0 and not self.bm25.corpus:
            log.info("Rebuilding BM25 from FAISS metadata (missing bm25.jsonl).")
            self.bm25.rebuild_from(self.idx.iter_meta())
            self.bm25.save()

    # ── Lazy-load getters (no-op when eager) ──────────────────────────────

    def asr(self):
        if self._asr is None:
            self._asr = build_asr(self.cfg)
        return self._asr

    def summ(self):
        if self._summ is None:
            self._summ = Summarizer(self.cfg)
        return self._summ

    def emb(self):
        if self._emb is None:
            self._emb = Embedder(self.cfg)
        return self._emb

    def _release(self, attr: str) -> None:
        """Free a loaded model and clear CUDA memory in low_vram mode."""
        if not self.low_vram:
            return
        obj = getattr(self, attr, None)
        if obj is None:
            return
        setattr(self, attr, None)
        del obj
        _free_cuda()

    def ingest(self, audio_path: str | Path, force: bool = False) -> dict:
        audio_path = str(audio_path)
        stem = Path(audio_path).stem
        asr_cfg = self.cfg.raw["asr"]
        merge_cfg = asr_cfg.get("merge", {})
        dedup_cfg = asr_cfg.get("dedup", {})

        # 1. transcribe 
        print(f"  [ASR] transcribing {Path(audio_path).name} ...")
        raw_segs = self.asr().transcribe(audio_path)
        print(f"  [ASR] raw segments: {len(raw_segs)}")
        self._release("_asr")  # free VRAM for summarizer if low_vram

        # 2. Cleanup 
        cleaned = cleanup_segments(
            raw_segs,
            max_run=int(dedup_cfg.get("max_token_run", 2)),
            hallucination_ratio=float(dedup_cfg.get("max_token_repeat_ratio", 0.6)),
            adjacent_dup_ratio=float(dedup_cfg.get("min_similarity", 0.85)),
        )
        print(f"  [Cleanup] kept {len(cleaned)}/{len(raw_segs)} segments")

        # 3. Merge into retrieval-friendly passages
        passages = merge_passages(
            cleaned,
            target_words=int(merge_cfg.get("target_words", 60)),
            max_words=int(merge_cfg.get("max_words", 100)),
            max_seconds=float(merge_cfg.get("max_seconds", 30.0)),
            overlap_segments=int(merge_cfg.get("overlap_segments", 1)),
        )
        print(f"  [Merge] {len(passages)} passage(s)")

        # 4. Summarize from cleaned text (raw text is noisy)
        full_text = " ".join(s["text"] for s in cleaned).strip()
        print(f"  [Summarize] {len(full_text)} chars ...")
        summary = self.summ().summarize(full_text) if full_text else ""
        self._release("_summ")  # free VRAM for embedder if low_vram

        # 5. Persist transcript + summary
        t_path = Path(self.cfg.raw["paths"]["transcripts_dir"]) / f"{stem}.json"
        s_path = Path(self.cfg.raw["paths"]["summaries_dir"]) / f"{stem}.txt"
        t_path.write_text(
            json.dumps(
                {
                    "audio_file": audio_path,
                    "raw_segments": raw_segs,
                    "segments": cleaned,
                    "passages": passages,
                    "summary": summary,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        s_path.write_text(summary, encoding="utf-8")

        # 6. Re-ingest path: wipe previous vectors for this audio file
        if force:
            removed = self.idx.remove_by_audio(audio_path)
            if removed:
                print(f"  [Reindex] removed {removed} prior segment(s)")

        # 7. Embed + add to FAISS (fingerprint dedup is automatic)
        added = 0
        if passages:
            print(f"  [Embed] {len(passages)} passages ...")
            vecs = self.emb().encode_passages([p["text"] for p in passages])
            added = self.idx.add(vecs, passages, audio_path)
            # 8. Rebuild BM25 from current FAISS metadata so IDs stay aligned
            self.bm25.rebuild_from(self.idx.iter_meta())
            self.idx.save()
            self.bm25.save()
            print(
                f"  [Index] +{added} new, total={self.idx.ntotal}  "
                f"(BM25 corpus={len(self.bm25.corpus)})"
            )

        return {
            "raw_segments": raw_segs,
            "segments": cleaned,
            "passages": passages,
            "summary": summary,
            "transcript_path": str(t_path),
            "added": added,
        }

    def search(
        self,
        query: str,
        k: int | None = None,
        mode: SearchMode | None = None,
    ) -> list[tuple[float, SegmentMeta]]:
        """Search the index and return top-k matching segments."""
        search_cfg = self.cfg.raw.get("search", {})
        k = k or int(search_cfg.get("top_k", 5))
        mode = mode or search_cfg.get("mode", "hybrid")
        fetch_k = int(search_cfg.get("fetch_k", max(50, k * 10)))
        rrf_k = int(search_cfg.get("rrf_k", 60))

        if self.idx.ntotal == 0:
            return []

        if mode == "semantic":
            q_vec = self.emb().encode_query(query)
            return self.idx.search(q_vec, k=k)

        if mode == "lexical":
            bm_hits = self.bm25.search(query, k=k)
            return [
                (score, self.idx.meta_by_id[mid])
                for score, mid in bm_hits
                if mid in self.idx.meta_by_id
            ]

        if mode == "hybrid":
            # Semantic rankings
            q_vec = self.emb().encode_query(query)
            sem = self.idx.search(q_vec, k=fetch_k)
            # Lexical rankings
            lex = self.bm25.search(query, k=fetch_k)

            fused: dict[int, float] = {}
            for rank, (_, m) in enumerate(sem):
                fused[m.id] = fused.get(m.id, 0.0) + 1.0 / (rrf_k + rank + 1)
            for rank, (_, mid) in enumerate(lex):
                fused[mid] = fused.get(mid, 0.0) + 1.0 / (rrf_k + rank + 1)

            ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
            return [
                (score, self.idx.meta_by_id[mid])
                for mid, score in ranked
                if mid in self.idx.meta_by_id
            ]

        raise ValueError(
            f"Unknown search mode '{mode}'. Use 'semantic', 'lexical', or 'hybrid' your mode is not available."
        )
