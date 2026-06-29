
from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from src.audio import slice_segment
from src.config import Config
from src.pipeline import Pipeline

CFG = Config.load()
PIPE = Pipeline(CFG)



def ui_ingest(audio_path: str | None, force: bool):
    if not audio_path:
        return "", "", pd.DataFrame(), pd.DataFrame()
    try:
        out = PIPE.ingest(audio_path, force=bool(force))
    except Exception as e:
        return f"Error: {e}", "", pd.DataFrame(), pd.DataFrame()

    transcript = "\n".join(s["text"] for s in out["segments"])

    seg_df = pd.DataFrame([
        {
            "start (s)": round(s["start"], 2),
            "end (s)": round(s["end"], 2),
            "text": s["text"],
        }
        for s in out["segments"]
    ])
    passage_df = pd.DataFrame([
        {
            "start (s)": round(p["start"], 2),
            "end (s)": round(p["end"], 2),
            "words": len(p["text"].split()),
            "text": p["text"],
        }
        for p in out["passages"]
    ])
    return transcript, out["summary"], seg_df, passage_df


def ui_search(query: str, k: int, mode: str):
    if not query or not query.strip():
        return pd.DataFrame(), None
    hits = PIPE.search(query, k=int(k), mode=mode)
    rows = [
        {
            "score": round(score, 4),
            "file": Path(m.audio_file).name,
            "start (s)": round(m.start, 2),
            "end (s)": round(m.end, 2),
            "text": m.text,
        }
        for score, m in hits
    ]
    preview_path = None
    if hits:
        _, top = hits[0]
        try:
            tmp = Path(tempfile.mkdtemp()) / "preview.wav"
            slice_segment(top.audio_file, top.start, top.end, tmp)
            preview_path = str(tmp)
        except Exception:
            pass
    return pd.DataFrame(rows), preview_path



DEFAULT_MODE = CFG.raw.get("search", {}).get("mode", "hybrid")
DEFAULT_K = CFG.raw.get("search", {}).get("top_k", 5)

with gr.Blocks(title="Arabic Audio Search") as demo:
    gr.Markdown(
        "# NLP project 2: Arabic Audio Understanding & Retrieval\n"
        "Upload Arabic audio to index it, then search across all indexed passages "
        "with semantic, lexical (BM25), or hybrid (RRF) retrieval.", 
        "Made by Amr Hussain under the supervison of Prof. Ahmed B. Zaky"
    )

    with gr.Tab("Ingest"):
        gr.Markdown(
            "Upload an Arabic audio file to transcribe, clean from dublications, merge into "
            "retrieval-friendly passages, summarize, and add to the search index."
        )
        audio_in = gr.Audio(type="filepath", label="Upload Arabic audio")
        force_cb = gr.Checkbox(
            label="Force re-ingest (remove prior segments for this file first)",
            value=False,
        )
        ingest_btn = gr.Button("Ingest", variant="primary")
        with gr.Row():
            transcript_box = gr.Textbox(label="Cleaned transcript", lines=8, max_lines=20)
            summary_box = gr.Textbox(label="Summary", lines=4)
        segments_df = gr.Dataframe(label="Cleaned segments (raw + dedup)", wrap=True)
        passages_df = gr.Dataframe(label="Indexed passages (merged)", wrap=True)
        ingest_btn.click(
            ui_ingest,
            inputs=[audio_in, force_cb],
            outputs=[transcript_box, summary_box, segments_df, passages_df],
        )

    with gr.Tab("Search"):
        gr.Markdown("Enter an Arabic query to find matching audio passages.")
        with gr.Row():
            query_box = gr.Textbox(label="Arabic query (استعلام)", scale=4)
            k_slider = gr.Slider(1, 20, value=DEFAULT_K, step=1, label="top-k", scale=1)
            mode_dd = gr.Dropdown(
                choices=["hybrid", "semantic", "lexical"],
                value=DEFAULT_MODE,
                label="Mode",
                scale=1,
            )
        search_btn = gr.Button("Search", variant="primary")
        results_df = gr.Dataframe(label="Hits", wrap=True)
        preview_audio = gr.Audio(label="Top hit preview", type="filepath")
        search_btn.click(
            ui_search,
            inputs=[query_box, k_slider, mode_dd],
            outputs=[results_df, preview_audio],
        )

if __name__ == "__main__":
    demo.launch()
