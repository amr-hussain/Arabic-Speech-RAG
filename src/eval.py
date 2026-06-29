from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def _plot_asr(result: dict, out_path: str = "eval_asr.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["WER", "CER"]
    values  = [result["wer"], result["cer"]]
    colors  = ["#e74c3c", "#e67e22"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(metrics, values, color=colors, width=0.4, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.2%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.3 + 0.05)
    ax.set_ylabel("Error Rate")
    ax.set_title(
        f"ASR Evaluation — FLEURS Arabic (ar_eg)\n"
        f"Model: Whisper-large-v3-turbo  |  n={result['n']} samples",
        fontsize=10,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {out_path}")


def _plot_summ(result: dict, out_path: str = "eval_summ.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    values  = [result["rouge1"], result["rouge2"], result["rougeL"]]
    colors  = ["#2ecc71", "#27ae60", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(metrics, values, color=colors, width=0.4, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.3 + 0.02)
    ax.set_ylabel("Score")
    ax.set_title(
        f"Summarization Evaluation — XL-Sum Arabic\n"
        f"Model: mT5-XLSum  |  n={result['n']} samples",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {out_path}")


def _plot_retrieval(result: dict, out_path: str = "eval_retrieval.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K_VALUES = [1, 3, 5]
    precision = [result[f"precision_at_{k}"] for k in K_VALUES]
    recall    = [result[f"recall_at_{k}"]    for k in K_VALUES]
    x = np.arange(len(K_VALUES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - width / 2, precision, width, label="Precision@K", color="#3498db", edgecolor="white")
    b2 = ax.bar(x + width / 2, recall,    width, label="Recall@K",    color="#9b59b6", edgecolor="white")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_VALUES])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title(
        f"Retrieval Evaluation (hybrid search)\n"
        f"MRR@5={result['mrr_at_5']:.4f}  |  n={result['n']} queries",
        fontsize=10,
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {out_path}")


# ── Evaluation functions ──────────────────────────────────────────────────────

def eval_asr(cfg, n: int = 100, plot: bool = True) -> dict:
    """Evaluate ASR on FLEURS Arabic and return WER/CER.

    Dataset: google/fleurs, config ar_eg (Egyptian Arabic), test split.
    ~428 test samples total; freely available on HuggingFace, no token required.
    Mozilla Common Voice moved off HuggingFace in October 2025.
    """
    import jiwer
    from datasets import load_dataset
    from src.asr import build_asr

    print("\n" + "=" * 60)
    print("Dataset: google/fleurs  |  config: ar_eg (Egyptian Arabic)")
    print("Split  : test  |  Metric: WER, CER")
    print("=" * 60)
    print(f"[eval_asr] Loading FLEURS Arabic (n={n}) ...")

    try:
        ds = load_dataset(
            "google/fleurs",
            "ar_eg",
            split=f"test[:{n}]",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Dataset load failed: {e}")
        return {"wer": None, "cer": None, "n": 0}

    asr = build_asr(cfg)
    refs, hyps = [], []

    for i, item in enumerate(ds):
        ref = item["transcription"].strip()
        audio_arr = item["audio"]["array"].astype(np.float32)
        sr = item["audio"]["sampling_rate"]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import soundfile as sf
            import librosa
            if sr != 16000:
                audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
            sf.write(tmp.name, audio_arr, 16000)
            tmp_path = tmp.name

        try:
            segs = asr.transcribe(tmp_path)
            hyp = " ".join(s["text"] for s in segs).strip()
        except Exception:
            hyp = ""
        finally:
            os.unlink(tmp_path)

        refs.append(ref)
        hyps.append(hyp)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(ds)} done ...")

    if not refs:
        print("  No valid reference samples found.")
        return {"wer": None, "cer": None, "n": 0}

    # jiwer requires the transform chain to TERMINATE in
    # ReduceToListOfListOfWords (for WER) or ReduceToListOfListOfChars (for CER)
    # so the output is list[list[str]] — without it, jiwer raises
    # "each reference should be a non-empty list of strings...".
    wer_transform = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    cer_transform = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ])

    # Filter pairs using jiwer's own word-level transform:
    #   - empty raw ref or empty post-transform ref → drop (bad annotation)
    #   - empty hyp → substitute placeholder so it still contributes as ~100% WER
    PLACEHOLDER_HYP = "unknowntoken"
    valid_refs, valid_hyps, empty_hyp_count = [], [], 0
    for ref, hyp in zip(refs, hyps):
        if not ref:
            continue
        ref_words = wer_transform([ref])[0]
        if not ref_words:
            continue
        hyp_words = wer_transform([hyp])[0] if hyp else []
        if not hyp_words:
            hyp = PLACEHOLDER_HYP
            empty_hyp_count += 1
        valid_refs.append(ref)
        valid_hyps.append(hyp)

    if not valid_refs:
        print("  No valid reference samples after normalization.")
        return {"wer": None, "cer": None, "n": 0}
    if empty_hyp_count:
        print(f"  Note: {empty_hyp_count} empty hypothesis/es replaced with placeholder "
              f"(counted as ~100% WER for those clips).")

    wer = jiwer.wer(valid_refs, valid_hyps,
                    reference_transform=wer_transform,
                    hypothesis_transform=wer_transform)
    cer = jiwer.cer(valid_refs, valid_hyps,
                    reference_transform=cer_transform,
                    hypothesis_transform=cer_transform)
    result = {"wer": round(wer, 4), "cer": round(cer, 4), "n": len(valid_refs)}

    print(f"\n[ASR Evaluation Results]  n={result['n']}")
    print(f"  WER: {result['wer']:.2%}")
    print(f"  CER: {result['cer']:.2%}")

    if plot:
        _plot_asr(result)
    return result


class _ArabicRougeTokenizer:
    """ROUGE tokenizer that preserves Arabic characters.

    rouge_score's default tokenizer does re.sub(r'[^a-z0-9]+', ' ', text)
    which strips *all* Arabic text, producing zero ROUGE scores.
    We reuse src.arabic_text.normalize_arabic for NFC + tashkeel/tatweel
    stripping + alef/ya/ta-marbuta unification, then Unicode-aware word split.
    """

    def __init__(self):
        import re
        from src.arabic_text import normalize_arabic
        self._re = re
        self._normalize = normalize_arabic
        self._punct = re.compile(r"[^\w\s]", flags=re.UNICODE)

    def tokenize(self, text: str) -> list[str]:
        text = self._normalize(text)
        text = self._punct.sub(" ", text)
        return [t for t in text.split() if t]


def eval_summ(cfg, n: int = 50, plot: bool = True) -> dict:
    """Evaluate summarization ROUGE on XL-Sum Arabic.

    Dataset: csebuetnlp/xlsum, config arabic, validation split.
    ~6,093 validation samples total; news article → headline pairs.
    """
    import numpy as _np
    from datasets import load_dataset
    from rouge_score import rouge_scorer
    from src.summarize import Summarizer

    print("\n" + "=" * 60)
    print("Dataset: csebuetnlp/xlsum  |  config: arabic")
    print("Split  : validation  |  Metric: ROUGE-1/2/L (Arabic-aware tokenizer)")
    print("=" * 60)
    print(f"[eval_summ] Loading XL-Sum Arabic (n={n}) ...")

    try:
        ds = load_dataset("csebuetnlp/xlsum", "arabic", split=f"validation[:{n}]")
    except Exception as e:
        print(f"  Dataset load failed: {e}")
        return {"rouge1": None, "rouge2": None, "rougeL": None, "n": 0}

    summ = Summarizer(cfg)
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=_ArabicRougeTokenizer(),
    )
    r1, r2, rL = [], [], []
    preds: list[str] = []

    for i, item in enumerate(ds):
        text = item["text"].strip()
        ref  = item["summary"].strip()
        # XL-Sum references are single-sentence headlines. Use the model's
        # direct single-shot path (not hierarchical join) for a fair
        # comparison — otherwise multi-paragraph summaries tank precision.
        pred = summ._summarize_once(text) if text else ""
        preds.append(pred)
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rL.append(scores["rougeL"].fmeasure)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(ds)} done ...")

    result = {
        "rouge1": round(float(_np.mean(r1)), 4),
        "rouge2": round(float(_np.mean(r2)), 4),
        "rougeL": round(float(_np.mean(rL)), 4),
        "n": len(preds),
    }
    print(f"\n[Summarization Evaluation Results]  n={result['n']}")
    print(f"  ROUGE-1: {result['rouge1']:.4f}")
    print(f"  ROUGE-2: {result['rouge2']:.4f}")
    print(f"  ROUGE-L: {result['rougeL']:.4f}")

    if plot:
        _plot_summ(result)
    return result


def eval_retrieval(
    cfg,
    queries_path: str = "tests/retrieval_queries.json",
    mode: str | None = None,
    plot: bool = True,
) -> dict:
    """Evaluate retrieval quality on a hand-written query set.

    Dataset: custom hand-written queries in tests/retrieval_queries.json.
    Each query specifies expected_audio and/or expected_text_substr.
    Assumes single-relevant-document per query (standard IR setting).
    """
    from src.pipeline import Pipeline

    print("\n" + "=" * 60)
    print(f"Dataset: {queries_path}  (hand-written queries)")
    print("Metrics: MRR@5, Precision@K, Recall@K  (K ∈ 1, 3, 5)")
    print("=" * 60)

    qp = Path(queries_path)
    if not qp.exists():
        print(f"Query file not found: {queries_path}")
        print("Create tests/retrieval_queries.json with your hand-written queries.")
        return {"mrr_at_5": None, "top1_hit_rate": None, "n": 0}

    queries = json.loads(qp.read_text(encoding="utf-8"))
    pipe = Pipeline(cfg)

    if pipe.idx.ntotal == 0:
        print("Index is empty. Run 'python -m scripts.ingest data/raw/' first.")
        return {"mrr_at_5": None, "top1_hit_rate": None, "n": 0}

    K_VALUES = [1, 3, 5]
    max_k = max(K_VALUES)

    rr_scores: list[float] = []
    hit_at: dict[int, list[int]] = {k: [] for k in K_VALUES}

    for item in queries:
        query = item["query"]
        exp_audio = item.get("expected_audio", "")
        exp_substr = item.get("expected_text_substr", "")
        hits = pipe.search(query, k=max_k, mode=mode)

        found_rank: int | None = None
        for rank, (score, m) in enumerate(hits, 1):
            hit_audio = Path(m.audio_file).name
            audio_match = (not exp_audio) or (exp_audio in hit_audio)
            text_match  = (not exp_substr) or (exp_substr in m.text)
            if audio_match and text_match:
                found_rank = rank
                break

        rr_scores.append(1.0 / found_rank if found_rank else 0.0)
        for k in K_VALUES:
            hit_at[k].append(1 if (found_rank is not None and found_rank <= k) else 0)

    mrr = float(np.mean(rr_scores)) if rr_scores else 0.0
    result: dict = {
        "mrr_at_5": round(mrr, 4),
        "top1_hit_rate": round(float(np.mean(hit_at[1])), 4) if hit_at[1] else 0.0,
        "n": len(queries),
    }
    for k in K_VALUES:
        recall_k    = float(np.mean(hit_at[k])) if hit_at[k] else 0.0
        precision_k = recall_k / k
        result[f"recall_at_{k}"]    = round(recall_k, 4)
        result[f"precision_at_{k}"] = round(precision_k, 4)

    print(f"\n[Retrieval Evaluation Results]  n={result['n']}")
    print(f"  MRR@5:          {result['mrr_at_5']:.4f}")
    print(f"  {'K':>3}  {'Precision@K':>12}  {'Recall@K':>10}")
    for k in K_VALUES:
        print(f"  {k:>3}  {result[f'precision_at_{k}']:>12.4f}  {result[f'recall_at_{k}']:>10.4f}")

    if plot:
        _plot_retrieval(result)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluation harness for the Arabic audio pipeline.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_asr = sub.add_parser("asr", help="ASR WER/CER evaluation (FLEURS Arabic)")
    p_asr.add_argument("--n", type=int, default=100, help="Number of FLEURS test clips")
    p_asr.add_argument("--no-plot", action="store_true", help="Skip matplotlib chart")
    p_asr.add_argument("--config", default="configs/config.yaml")

    p_summ = sub.add_parser("summ", help="Summarization ROUGE evaluation")
    p_summ.add_argument("--n", type=int, default=50, help="Number of XL-Sum samples")
    p_summ.add_argument("--no-plot", action="store_true", help="Skip matplotlib chart")
    p_summ.add_argument("--config", default="configs/config.yaml")

    p_ret = sub.add_parser("retrieval", help="Retrieval MRR/Precision/Recall evaluation")
    p_ret.add_argument("--queries", default="tests/retrieval_queries.json")
    p_ret.add_argument(
        "--mode",
        choices=("semantic", "lexical", "hybrid"),
        default=None,
        help="Search mode (default: config.search.mode)",
    )
    p_ret.add_argument("--no-plot", action="store_true", help="Skip matplotlib chart")
    p_ret.add_argument("--config", default="configs/config.yaml")

    args = ap.parse_args()

    from src.config import Config
    cfg = Config.load(args.config)

    if args.cmd == "asr":
        eval_asr(cfg, n=args.n, plot=not args.no_plot)
    elif args.cmd == "summ":
        eval_summ(cfg, n=args.n, plot=not args.no_plot)
    elif args.cmd == "retrieval":
        eval_retrieval(cfg, queries_path=args.queries, mode=args.mode, plot=not args.no_plot)


if __name__ == "__main__":
    main()
