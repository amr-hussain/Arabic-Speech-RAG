from __future__ import annotations

import argparse

from src.config import Config
from src.pipeline import Pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Search indexed Arabic audio segments.")
    ap.add_argument("query", help="Arabic text query")
    ap.add_argument("-k", type=int, default=None, help="Number of results (default: config top_k)")
    ap.add_argument(
        "--mode",
        choices=("semantic", "lexical", "hybrid"),
        default=None,
        help="Search mode (default: config.search.mode, typically 'hybrid')",
    )
    ap.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    pipe = Pipeline(cfg)

    if pipe.idx.ntotal == 0:
        print("Index is empty. Run 'python -m scripts.ingest <path>' first.")
        return

    hits = pipe.search(args.query, k=args.k, mode=args.mode)
    mode_used = args.mode or cfg.raw.get("search", {}).get("mode", "hybrid")
    if not hits:
        print("No results found.")
        return

    print(f"\nQuery: {args.query}    [mode={mode_used}]\n{'─' * 60}")
    for rank, (score, m) in enumerate(hits, 1):
        print(
            f"#{rank}  score={score:.3f}  [{m.start:.1f}s – {m.end:.1f}s]"
            f"\n    file: {m.audio_file}"
            f"\n    text: {m.text[:200]}\n"
        )


if __name__ == "__main__":
    main()
