# Arabic Audio Understanding & Retrieval System

A deep-learning pipeline that converts Arabic speech to text, summarizes the content, and enables semantic search over audio recordings.

## Models Used

| Stage | Model | Size | Notes |
|---|---|---|---|
| ASR (default) | `openai/whisper-v3-turbo` | 244 M | Forced `language=ar`; runs locally on ≥2 GB VRAM |
| ASR (fallback) | `openai/whisper-large-v3` | API | HF Inference API; set `asr.backend: hf_api` |
| Summarization | `csebuetnlp/mT5_multilingual_XLSum` | 580 M | Trained on multilingual news incl. Arabic |
| Embeddings | `intfloat/multilingual-e5-base` | 278 M | Requires `"query:"` / `"passage:"` prefixes |
| Vector store | FAISS `IndexFlatIP` | — | hybrid bm25 + semantic search via |


# Set up environment variables
cp .env.example .env
# Edit .env and paste your HuggingFace token (needed for CV dataset + large-v3 API)


## Ingest Audio

```bash
# Single file
python -m scripts.ingest data/raw/lecture.wav

# Entire directory (wav, mp3, flac, m4a, ogg)
python -m scripts.ingest data/raw/ --force #to delete the voice embeddings before causing dublication
```

Output is written to:
- `data/transcripts/<name>.json` — full transcript with per-segment timestamps
- `data/summaries/<name>.txt` — Arabic summary
- `data/index/` — FAISS index + metadata (auto-updated)

## Search

```bash
python -m scripts.search "aracib text "
```

## Gradio Web App

```bash
python app.py
```

## Evaluation

**ASR — WER/CER on Common Voice Arabic:**
```bash
python -m src.eval asr --n 100
```

**Summarization — ROUGE on XL-Sum Arabic:**
```bash
python -m src.eval summ --n 50
```

**Retrieval — MRR@5 on a hand-written query set:**
```bash
# First edit tests/retrieval_queries.json with your own queries and expected matches
python -m src.eval retrieval --queries tests/retrieval_queries.json
```

## Datasets

| Purpose | Dataset | Link |
|---|---|---|
| ASR evaluation | Mozilla Common Voice 17 (Arabic) | https://commonvoice.mozilla.org/ar |
| Summarization evaluation | XL-Sum (Arabic) | https://huggingface.co/datasets/csebuetnlp/xlsum |
| Semantic search evaluation | Hand-written query set | `tests/retrieval_queries.json` |
| ASR training data (reference) | MASC Arabic Speech | https://huggingface.co/datasets/hirundo-io/MASC |


source /mnt/D/pip_envs/asr/bin/activate

# 1. Put Arabic audio in data/raw/, then ingest:
python -m scripts.ingest data/raw/

# 2. Search:
python -m scripts.search "text arabic"

# 3. Launch Gradio app:
python app.py

# 4. Evaluate:
python -m src.eval asr --n 20
python -m src.eval summ --n 10

To switch to whisper-large-v3 (HF API): set asr.backend: hf_api in
configs/config.yaml and add HF_TOKEN to .env.