from __future__ import annotations

import hashlib
import re
import unicodedata #used for normalization
from difflib import SequenceMatcher

# Tashkeel (harakat)
_TASHKEEL_RE = re.compile(
    r"[\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08E1\u08E3-\u08FF]"
)
# Tatweel (kashida)
_TATWEEL = "\u0640"
# Zero-width / directional characters that creep in from copy-paste
_ZW_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]")

# Punctuation (Arabic + Latin) to strip for tokenization/fingerprinting
_PUNCT_RE = re.compile(
    r"[\u0600-\u0605\u060C-\u060F\u061B\u061E\u061F\u066A-\u066D\u06D4"
    r"!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؛؟«»—–…]"
)

# Arabic letter regex (for tokenization)
_AR_TOKEN_RE = re.compile(r"[\u0621-\u064A\u0660-\u0669a-zA-Z0-9]+")

# Character equivalence classes for search/fingerprint normalization.
#   Alef family → bare alef
#   Ya family   → ya
#   Ta-marbuta  → ha (common search normalization)
#   Waw with hamza → waw
#   Ya with hamza  → ya
_NORMALIZE_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي",
    "ؤ": "و",
    "ة": "ه",
    "ـ": "",  # tatweel fallback (also stripped below)
})


def strip_diacritics(text: str) -> str:
    """Remove tashkeel, tatweel, and zero-width marks."""
    text = _TASHKEEL_RE.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = _ZW_RE.sub("", text)
    return text


def normalize_arabic(text: str) -> str:
    """
    Canonicalize Arabic for search and fingerprinting.

    Steps:
      1. NFC Unicode composition
      2. strip tashkeel + tatweel + zero-width
      3. collapse alef / ya / ta-marbuta / waw-hamza variants
      4. lowercase Latin
      5. collapse runs of whitespace
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = strip_diacritics(text)
    text = text.translate(_NORMALIZE_MAP)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """
    Arabic-aware tokenization for BM25.

    Normalizes the text, then extracts alphanumeric Arabic + Latin tokens.
    Single-character tokens are kept (proper nouns can be short).
    """
    return _AR_TOKEN_RE.findall(normalize_arabic(text))


def collapse_repeats(text: str, max_run: int = 2) -> str:
    """
    Collapse runs of an identical token repeated more than `max_run` times.

    Example (max_run=2):
        "آه آه آه آه طيب طيب"  →  "آه آه طيب طيب"

    """
    if not text:
        return text
    tokens = text.split()
    if len(tokens) < max_run + 1:
        return text
    out: list[str] = []
    run_token: str | None = None
    run_count = 0
    for tok in tokens:
        key = normalize_arabic(tok)
        if key == run_token:
            run_count += 1
            if run_count <= max_run:
                out.append(tok)
        else:
            run_token = key
            run_count = 1
            out.append(tok)
    return " ".join(out)


def looks_like_hallucination(text: str, max_ratio: float = 0.6) -> bool:
    """
    Heuristic: a Whisper segment is likely a hallucination if one token
    dominates (≥ max_ratio of all tokens) and there are at least a few tokens.

    Catches classic failure modes like:
        "شكرا شكرا شكرا شكرا شكرا شكرا شكرا"
        "نعم نعم نعم نعم نعم نعم"
    """
    toks = tokenize(text)
    if len(toks) < 4:
        return False
    # Count most frequent token
    from collections import Counter
    top, count = Counter(toks).most_common(1)[0]
    return (count / len(toks)) >= max_ratio


def similarity_ratio(a: str, b: str) -> float:
    """Cheap fuzzy similarity ratio (0..1) over normalized Arabic text."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_arabic(a), normalize_arabic(b)).ratio()


def content_fingerprint(text: str, audio_file: str, start: float) -> str:
    """
    Stable content fingerprint for deduplication.

    Combines (normalized text, audio path, rounded start timestamp) so that
    re-ingesting the same file produces identical fingerprints and new
    passages from a different file don't collide by text alone.
    """
    key = f"{normalize_arabic(text)}|{audio_file}|{round(float(start), 1)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()
