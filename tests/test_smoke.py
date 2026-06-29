import numpy as np
import pytest

from src.arabic_text import (
    collapse_repeats,
    content_fingerprint,
    looks_like_hallucination,
    normalize_arabic,
    similarity_ratio,
    strip_diacritics,
    tokenize,
)
from src.asr import cleanup_segments, merge_passages
from src.audio import AudioChunk, chunk
from src.config import Config


def test_chunk_monotonic():
    y = np.zeros(16000 * 10, dtype=np.float32)
    chunks = chunk(y, chunk_s=4.0, overlap_s=1.0)
    assert len(chunks) >= 2
    assert chunks[0].start_s == 0.0
    assert chunks[-1].end_s == pytest.approx(10.0, abs=0.01)
    for c in chunks:
        assert isinstance(c, AudioChunk)
        assert c.samples.dtype == np.float32


def test_chunk_short_audio():
    y = np.zeros(16000 * 2, dtype=np.float32)
    chunks = chunk(y, chunk_s=30.0, overlap_s=1.0)
    assert len(chunks) == 1
    assert chunks[0].start_s == 0.0


def test_config_loads():
    cfg = Config.load()
    assert cfg.device in ("cuda", "cpu")
    assert "asr" in cfg.raw
    assert "summarize" in cfg.raw
    assert "embed" in cfg.raw
    assert "index" in cfg.raw
    assert "search" in cfg.raw
    # New mandatory sections
    assert "merge" in cfg.raw["asr"]
    assert "dedup" in cfg.raw["asr"]
    assert cfg.raw["search"].get("mode") in ("semantic", "lexical", "hybrid")


def test_strip_diacritics_removes_tashkeel():
    # "مُحَمَّدٌ" → "محمد"
    assert strip_diacritics("مُحَمَّدٌ") == "محمد"
    # Tatweel should be stripped too
    assert strip_diacritics("سـلـام") == "سلام"


def test_normalize_arabic_folds_variants():
    assert normalize_arabic("أحمد") == normalize_arabic("احمد")
    assert normalize_arabic("إيمان") == normalize_arabic("ايمان")
    assert normalize_arabic("مدرسة") == normalize_arabic("مدرسه")
    # Case + whitespace
    assert normalize_arabic("  Hello  WORLD  ") == "hello world"


def test_tokenize_arabic():
    toks = tokenize("السلام عليكم، كيف حالك؟")
    assert "السلام" in toks
    assert "عليكم" in toks
    assert "،" not in toks


def test_collapse_repeats_trims_runs():
    assert collapse_repeats("آه آه آه آه طيب") == "آه آه طيب"
    # With tashkeel the two tokens still match after normalization
    assert collapse_repeats("نَعَم نعم نعم").split().count("نعم") <= 2


def test_collapse_repeats_preserves_non_dup():
    s = "اهلا وسهلا بكم في الجامعة"
    assert collapse_repeats(s) == s


def test_hallucination_detector():
    assert looks_like_hallucination("شكرا شكرا شكرا شكرا شكرا شكرا شكرا")
    assert not looks_like_hallucination("اليوم ذهبت الى الجامعة لحضور المحاضرة")


def test_similarity_ratio_bounds():
    assert similarity_ratio("السلام عليكم", "السلام عليكم") == pytest.approx(1.0)
    assert 0.0 <= similarity_ratio("محاضرة", "اقتصاد") < 0.5


def test_fingerprint_stable_and_sensitive():
    a = content_fingerprint("مرحبا", "x.wav", 1.0)
    b = content_fingerprint("مرحبا", "x.wav", 1.04)      # rounds to 1.0
    c = content_fingerprint("مرحبا", "x.wav", 1.2)       # different second
    d = content_fingerprint("مرحبا", "y.wav", 1.0)       # different file
    assert a == b
    assert a != c
    assert a != d


def _seg(start, end, text):
    return {"start": float(start), "end": float(end), "text": text}


def test_cleanup_drops_hallucinations_and_adjacent_dups():
    segs = [
        _seg(0, 2, "شكرا لكم على الحضور"),
        _seg(2, 4, "شكرا شكرا شكرا شكرا شكرا شكرا"),          # hallucination
        _seg(4, 6, "شكرا لكم على الحضور"),                    # adjacent dup
        _seg(6, 9, "اليوم سنتحدث عن الاقتصاد الكلي"),
    ]
    out = cleanup_segments(segs)
    texts = [s["text"] for s in out]
    assert len(out) == 2
    assert any("الاقتصاد" in t for t in texts)
    # End of the adjacent-dup should extend the first segment's end
    assert out[0]["end"] >= 6.0


def test_merge_passages_combines_short_segments():
    # 10 short segments, each 3 words → should coalesce to ≤ 10/5 = 2-ish passages
    segs = [
        _seg(i * 2, i * 2 + 2, f"كلمة{i}_أ كلمة{i}_ب كلمة{i}_ج")
        for i in range(10)
    ]
    passages = merge_passages(
        segs,
        target_words=15,
        max_words=30,
        max_seconds=100.0,
        overlap_segments=0,
    )
    assert passages, "merger should produce at least one passage"
    for p in passages:
        assert len(p["text"].split()) <= 30
        assert p["start"] <= p["end"]
    # Start/end of the merged passages should cover the full time range
    assert passages[0]["start"] == 0.0
    assert passages[-1]["end"] == pytest.approx(20.0, abs=0.01)


def test_merge_passages_respects_max_seconds():
    # Segments that are far apart in time should split even if word count is low
    segs = [
        _seg(0, 5, "مقدمة قصيرة عن الدرس"),
        _seg(90, 95, "نهاية الدرس"),
    ]
    passages = merge_passages(
        segs,
        target_words=50,
        max_words=100,
        max_seconds=30.0,
        overlap_segments=0,
    )
    assert len(passages) == 2


def test_merge_overlap_adds_tail():
    segs = [_seg(i, i + 1, f"كلمة{i}") for i in range(6)]
    no_overlap = merge_passages(
        segs, target_words=2, max_words=4, max_seconds=60.0, overlap_segments=0
    )
    with_overlap = merge_passages(
        segs, target_words=2, max_words=4, max_seconds=60.0, overlap_segments=1
    )
    # With overlap, total passage count is ≥ without overlap
    assert len(with_overlap) >= len(no_overlap)


def test_rrf_fusion_formula():
    # Minimal local RRF check: the fusion formula used by Pipeline.search
    # 1/(rrf_k + rank + 1) prefers items ranked high by either ranker.
    sem_ranks = [(0.9, 10), (0.8, 11), (0.7, 12)]
    lex_ranks = [(5.0, 12), (4.0, 13)]
    rrf_k = 60
    fused: dict[int, float] = {}
    for rank, (_, mid) in enumerate(sem_ranks):
        fused[mid] = fused.get(mid, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, (_, mid) in enumerate(lex_ranks):
        fused[mid] = fused.get(mid, 0.0) + 1.0 / (rrf_k + rank + 1)
    # Item 12 appears in both rankers → should beat 10 (which only appears once)
    assert fused[12] > fused[10]
    # Item 13 appears only in lexical with rank 1 → should be below 10
    assert fused[13] < fused[10]
