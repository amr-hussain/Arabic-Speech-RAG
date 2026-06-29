from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Config
from src.pipeline import Pipeline

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".webm"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Arabic audio into the search index.")
    ap.add_argument("path", help="Audio file or directory to ingest")
    ap.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Remove any existing segments for the audio file before re-indexing.",
    )
    args = ap.parse_args()

    cfg = Config.load(args.config)
    pipe = Pipeline(cfg)

    p = Path(args.path)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
    else:
        print(f"Error: '{p}' is not a file or directory.")
        return

    if not files:
        print(f"No audio files found in '{p}'.")
        return

    print(f"Ingesting {len(files)} file(s)...\n")
    for f in files:
        print(f"[ingest] {f}")
        try:
            out = pipe.ingest(f, force=args.force)
            print(
                f"  -> passages={len(out['passages'])} "
                f"(raw={len(out['raw_segments'])}, "
                f"cleaned={len(out['segments'])})  "
                f"added={out['added']}  "
                f"summary_chars={len(out['summary'])}  "
                f"transcript={out['transcript_path']}\n"
            )
        except Exception as e:
            print(f"  ERROR: {e}\n")

    print("Done.")


if __name__ == "__main__":
    main()
