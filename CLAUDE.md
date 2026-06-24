# unlimited-ocr-archive-test

Run two open-weights OCR engines over four real archival documents (2 VVA,
2 ASYOUWERE) and compare them against the API-grade baselines already in
hand. Engines tested:

- **Unlimited-OCR** (`baidu/Unlimited-OCR`, 3 B VLM, MIT, Nov 2026)
- **NuExtract 3** (`numind/NuExtract3`, 4 B VLM on Qwen3.5-VL, MIT) — called
  in OCR mode with template `{"full_text": "verbatim-string"}`

Baselines already committed:

- **Claude Sonnet 4.5 vision** for VVA (handwritten / typewritten letters)
- **docling** for ASYOUWERE (1919 multi-column newsprint)

Built to share publicly — keep the surface small.

## Stack

- Python 3.12
- [uv](https://docs.astral.sh/uv/) — package and project manager
- PEP 723 single-file script — `unlimited_ocr_test.py` declares its own deps
- `transformers`, `torch` (BF16 + CUDA), `pymupdf` (PDF → PNG), `python-dotenv`

## Layout

```
docs/<src>/*.pdf                       # 4 source docs, committed
baselines/<src>/*.{txt,md,meta.json}   # prior-engine OCR ground truth, committed
outputs/<src>/*.unlimited-ocr.{txt,meta.json}   # populated by unlimited_ocr_test.py
outputs/<src>/*.nuextract3.{txt,meta.json}      # populated by nuextract3_test.py
outputs/_raw/<stem>/                   # raw model output (audit trail) — both engines
outputs/comparison.md                  # built by compare.py (3-engine table)
outputs/summary.json                   # Unlimited-OCR summary
outputs/summary_nuextract3.json        # NuExtract 3 summary
```

## Hard constraints

- **CUDA required** — Unlimited-OCR is BF16 on NVIDIA. The script exits 1 on a
  Mac. To run from this Mac, push to GitHub and run on Colab/EC2/RunPod.
- **Per-item resumable** — outputs are checked before each PDF is rasterised.
  Add `--force` to re-run.
- **Live progress** — every doc prints start + finish with elapsed time. No
  silent multi-minute waits.

## Environment Setup

```bash
cp .env.example .env
# HF_TOKEN required (the model is gated only by HF rate limits for anon downloads)
```

## Development Commands

```bash
uv sync                              # install deps
uv run unlimited_ocr_test.py         # Unlimited-OCR, Gundam mode (default)
uv run unlimited_ocr_test.py --mode base
uv run unlimited_ocr_test.py --src vva
uv run unlimited_ocr_test.py --force
uv run nuextract3_test.py            # NuExtract 3 with {"full_text": "verbatim-string"}
uv run nuextract3_test.py --src vva
uv run nuextract3_test.py --force
uv run compare.py                    # build outputs/comparison.md
```

## Conventions

- Output filenames carry the engine name: `<stem>.unlimited-ocr.txt`,
  `<stem>.nuextract3.txt`, `<stem>.claude-sonnet-4-5.txt`, `<stem>.docling.txt`.
  Don't drop the engine suffix — the whole point of this repo is engine-to-engine
  comparison.
- `compare.py` reads from `baselines/` (committed) and `outputs/` (produced),
  not from anywhere else. To add a fourth engine: either ship a baseline file
  (update `BASELINE_ENGINES`) or write a `<engine>_test.py` that emits into
  `outputs/<src>/<stem>.<engine>.{txt,meta.json}` and add it to `ENGINE_LAYOUT`.
- NuExtract 3 is a structured extractor used here in OCR mode with the
  template `{"full_text": "verbatim-string"}`. Don't change the template
  without noting it in CLAUDE.md and the README — the apples-to-apples
  comparison story depends on a fixed template.
- Don't expand scope beyond OCR engine comparison. This repo's job is one
  question — "how does Unlimited-OCR look on these four docs?" — and being
  legible to a LinkedIn reader. RAG, retrieval, downstream NER all belong
  elsewhere.
