# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "torchvision",
#   "transformers>=4.49",
#   "Pillow",
#   "einops",
#   "addict",
#   "easydict",
#   "matplotlib",
#   "psutil",
#   "pymupdf",
#   "accelerate",
#   "rich",
#   "python-dotenv",
# ]
# ///
"""
unlimited_ocr_test.py — Run baidu/Unlimited-OCR over the docs/ sample set.

Walks every PDF under docs/<src>/ where <src> is vva or asyouwere. For each
PDF, rasterises pages at 300 DPI to PNG, runs Unlimited-OCR (Gundam mode by
default), and writes:

    outputs/<src>/<stem>.unlimited-ocr.txt        extracted text
    outputs/<src>/<stem>.unlimited-ocr.meta.json  model, mode, n_pages, elapsed, chars

Per-item resumable: if the output text file already exists, the PDF is skipped
(use --force to re-run). Progress is printed live, one line per doc.

Requires a CUDA GPU. Tested on Colab T4 (16 GB) — the 3B model fits in BF16.

Run:
    uv run unlimited_ocr_test.py                  # all docs, Gundam mode
    uv run unlimited_ocr_test.py --mode base      # base single-image mode
    uv run unlimited_ocr_test.py --src vva        # only VVA docs
    uv run unlimited_ocr_test.py --force          # ignore existing outputs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_ID = "baidu/Unlimited-OCR"


def rasterise_pdf(pdf_path: Path, dpi: int = 300) -> tuple[list[Path], Path]:
    """Render every page of a PDF to a PNG in a fresh temp dir.

    Returns (sorted page paths, temp dir to clean up later).
    """
    import fitz  # pymupdf

    tmp = Path(tempfile.mkdtemp(prefix=f"uocr_{pdf_path.stem}_"))
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pages: list[Path] = []
    for i, page in enumerate(doc):
        out = tmp / f"page_{i + 1:04d}.png"
        page.get_pixmap(matrix=mat).save(out)
        pages.append(out)
    doc.close()
    return pages, tmp


def load_model():
    """Load Unlimited-OCR in BF16 on CUDA. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        sys.exit(
            "ERROR: CUDA GPU not detected. Unlimited-OCR requires NVIDIA CUDA "
            "(BF16). Run this on a Colab T4/L4, an EC2 g5/g6, or any CUDA box."
        )

    print(f"Loading {MODEL_ID} (BF16) on {torch.cuda.get_device_name(0)}...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)
    return model, tokenizer


def collect_output_text(tmp_out: Path) -> str:
    """Unlimited-OCR writes its parsed text into output_path. Collect it.

    The exact filenames are not formally documented on the model card; we
    look for .md (preferred) or .txt files inside the output dir.
    """
    candidates = sorted(tmp_out.rglob("*.md")) + sorted(tmp_out.rglob("*.txt"))
    if not candidates:
        return ""
    return "\n\n".join(p.read_text(errors="replace") for p in candidates).strip()


def run_one_pdf(
    pdf_path: Path,
    model,
    tokenizer,
    mode: str,
    dpi: int,
) -> dict:
    """OCR a single PDF. Returns a metadata dict."""
    pages, tmp_in = rasterise_pdf(pdf_path, dpi=dpi)
    tmp_out = Path(tempfile.mkdtemp(prefix=f"uocr_out_{pdf_path.stem}_"))

    if mode == "gundam":
        kwargs = dict(base_size=1024, image_size=640, crop_mode=True)
    else:  # base
        kwargs = dict(base_size=1024, image_size=1024, crop_mode=False)

    t0 = time.time()
    if len(pages) == 1:
        model.infer(
            tokenizer,
            prompt="<image>document parsing.",
            image_file=str(pages[0]),
            output_path=str(tmp_out),
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=True,
            **kwargs,
        )
    else:
        model.infer_multi(
            tokenizer,
            prompt="<image>Multi page parsing.",
            image_files=[str(p) for p in pages],
            output_path=str(tmp_out),
            image_size=1024,
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            save_results=True,
        )
    elapsed = time.time() - t0

    text = collect_output_text(tmp_out)

    # Clean up temp page images; keep tmp_out under outputs/_raw/<stem>/ for audit
    raw_keep = OUTPUTS_DIR / "_raw" / pdf_path.stem
    raw_keep.mkdir(parents=True, exist_ok=True)
    for f in tmp_out.rglob("*"):
        if f.is_file():
            (raw_keep / f.name).write_bytes(f.read_bytes())
    for p in pages:
        try:
            p.unlink()
        except OSError:
            pass
    try:
        tmp_in.rmdir()
    except OSError:
        pass

    return {
        "model": MODEL_ID,
        "mode": mode,
        "n_pages": len(pages),
        "dpi": dpi,
        "elapsed_s": round(elapsed, 2),
        "char_count": len(text),
        "source_pdf": str(pdf_path.relative_to(PROJECT_ROOT)),
        "text": text,
    }


def write_outputs(out_dir: Path, stem: str, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = meta.pop("text")
    (out_dir / f"{stem}.unlimited-ocr.txt").write_text(text)
    (out_dir / f"{stem}.unlimited-ocr.meta.json").write_text(
        json.dumps(meta, indent=2)
    )


def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["gundam", "base"],
        default="gundam",
        help="Gundam = 1024 base / 640 image with crop (dense / large pages). "
        "Base = 1024/1024 without crop (single page). Default: gundam.",
    )
    ap.add_argument(
        "--src",
        choices=["vva", "asyouwere", "all"],
        default="all",
        help="Which doc folder to process. Default: all.",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-OCR even if an output file already exists.",
    )
    args = ap.parse_args()

    srcs = ["vva", "asyouwere"] if args.src == "all" else [args.src]
    pdfs: list[tuple[str, Path]] = []
    for src in srcs:
        src_dir = DOCS_DIR / src
        if not src_dir.exists():
            print(f"  skip: {src_dir} does not exist", flush=True)
            continue
        for pdf in sorted(src_dir.glob("*.pdf")):
            pdfs.append((src, pdf))

    if not pdfs:
        print("No PDFs found under docs/. Nothing to do.", flush=True)
        return 1

    print(f"Found {len(pdfs)} PDF(s) under docs/. Mode: {args.mode}.", flush=True)

    model, tokenizer = load_model()

    summary: list[dict] = []
    for i, (src, pdf) in enumerate(pdfs, 1):
        out_dir = OUTPUTS_DIR / src
        out_txt = out_dir / f"{pdf.stem}.unlimited-ocr.txt"
        if out_txt.exists() and not args.force:
            existing = json.loads(
                (out_dir / f"{pdf.stem}.unlimited-ocr.meta.json").read_text()
            )
            print(
                f"[{i}/{len(pdfs)}] {src}/{pdf.name} — already done "
                f"({existing.get('char_count', '?')} chars, "
                f"{existing.get('elapsed_s', '?')}s). Skipping.",
                flush=True,
            )
            summary.append({"src": src, "stem": pdf.stem, **existing})
            continue

        print(f"[{i}/{len(pdfs)}] {src}/{pdf.name} — running...", flush=True)
        meta = run_one_pdf(pdf, model, tokenizer, mode=args.mode, dpi=args.dpi)
        print(
            f"    done in {meta['elapsed_s']}s — {meta['n_pages']} page(s), "
            f"{meta['char_count']} chars",
            flush=True,
        )
        write_outputs(out_dir, pdf.stem, dict(meta))  # write_outputs pops 'text'
        summary.append(
            {"src": src, "stem": pdf.stem, **{k: v for k, v in meta.items() if k != "text"}}
        )

    (OUTPUTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Summary written to outputs/summary.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
