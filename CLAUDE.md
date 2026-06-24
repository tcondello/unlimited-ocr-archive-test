# unlimited-ocr-archive-test

Test Baidu's [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) 3B VLM
on four real archival documents (2 VVA, 2 ASYOUWERE) and compare against the
OCR engines previously run on the same docs (Claude Sonnet 4.5 vision for VVA,
docling for ASYOUWERE). Built to share publicly — keep the surface small.

## Stack

- Python 3.12
- [uv](https://docs.astral.sh/uv/) — package and project manager
- PEP 723 single-file script — `unlimited_ocr_test.py` declares its own deps
- `transformers`, `torch` (BF16 + CUDA), `pymupdf` (PDF → PNG), `python-dotenv`

## Layout

```
docs/<src>/*.pdf                       # 4 source docs, committed
baselines/<src>/*.{txt,md,meta.json}   # prior-engine OCR ground truth, committed
outputs/<src>/*.unlimited-ocr.{txt,meta.json}   # populated by the script
outputs/_raw/<stem>/                   # raw model output (audit trail)
outputs/comparison.md                  # built by compare.py
outputs/summary.json                   # built by unlimited_ocr_test.py
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
uv run unlimited_ocr_test.py         # Gundam mode (default)
uv run unlimited_ocr_test.py --mode base
uv run unlimited_ocr_test.py --src vva
uv run unlimited_ocr_test.py --force
uv run compare.py                    # build outputs/comparison.md
```

## Conventions

- Output filenames carry the engine name: `<stem>.unlimited-ocr.txt`,
  `<stem>.claude-sonnet-4-5.txt`, `<stem>.docling.txt`. Don't drop the engine
  suffix — the whole point of this repo is engine-to-engine comparison.
- `compare.py` reads from `baselines/` (committed) and `outputs/` (produced),
  not from anywhere else. If you add a fifth engine, add a `baselines/<src>/`
  file with the engine-suffixed name and update `BASELINE_ENGINES`.
- Don't expand scope beyond OCR engine comparison. This repo's job is one
  question — "how does Unlimited-OCR look on these four docs?" — and being
  legible to a LinkedIn reader. RAG, retrieval, downstream NER all belong
  elsewhere.
