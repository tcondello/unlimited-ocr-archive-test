# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "torchvision",
#   "transformers>=4.49",
#   "accelerate>=0.30",
#   "Pillow",
#   "pymupdf",
#   "python-dotenv",
# ]
# ///
"""
nuextract3_test.py — Run numind/NuExtract3 over the docs/ sample set.

NuExtract 3 is a 4B vision-language structured-extraction model (built on
Qwen3.5-VL). Strictly it's not an OCR engine — you give it a JSON template
and it returns structured fields. For an apples-to-apples OCR baseline we
ask for verbatim text with the template:

    {"full_text": "verbatim-string"}

Per page, the script calls the model, parses the JSON, and pulls
`full_text`. Pages are concatenated into one output, mirroring the format
the other engines produce.

For each PDF, writes:

    outputs/<src>/<stem>.nuextract3.txt        extracted text
    outputs/<src>/<stem>.nuextract3.meta.json  model, n_pages, elapsed, chars

Per-item resumable — skip if the .txt exists (use --force to override).
Live progress per doc + per page.

Requires a CUDA GPU. T4 works in fp16 (model is ~10 GB); A10/L4+ get bf16.

Run:
    uv run nuextract3_test.py
    uv run nuextract3_test.py --src vva
    uv run nuextract3_test.py --force
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_ID = "numind/NuExtract3"
TEMPLATE = {"full_text": "verbatim-string"}


def rasterise_pdf(pdf_path: Path, dpi: int = 300) -> tuple[list[Path], Path]:
    """Render every page of a PDF to PNG. Returns (page paths, temp dir)."""
    import fitz  # pymupdf

    tmp = Path(tempfile.mkdtemp(prefix=f"nu3_{pdf_path.stem}_"))
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
    """Load NuExtract 3. Returns (processor, model, dtype_label)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        sys.exit(
            "ERROR: CUDA GPU not detected. NuExtract 3 needs NVIDIA. "
            "Run this on a Colab T4 (fp16), L4 or A10+ (bf16), or any CUDA box."
        )

    device = "cuda"
    cap = torch.cuda.get_device_capability(0)
    # T4 / sm_75 has no hardware bf16 — fp16 there. Ampere+ gets bf16.
    dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    dtype_label = "bf16" if dtype is torch.bfloat16 else "fp16"

    print(
        f"Loading {MODEL_ID} ({dtype_label}) on {torch.cuda.get_device_name(0)}...",
        flush=True,
    )
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)
    return processor, model, dtype_label


def parse_full_text(response: str) -> str:
    """Pull the `full_text` field from NuExtract's JSON response.

    Tolerant of code fences and stray text around the JSON object.
    Falls back to the raw response if nothing parses.
    """
    if not response:
        return ""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if "```" in cleaned:
            cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    for candidate in (cleaned, _greedy_object(cleaned)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "full_text" in obj:
                value = obj["full_text"]
                return value if isinstance(value, str) else json.dumps(value)
        except json.JSONDecodeError:
            continue
    return response


def _greedy_object(text: str) -> str | None:
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        return text[first : last + 1]
    return None


def run_one_pdf(pdf_path: Path, processor, model, dpi: int) -> dict:
    """OCR a single PDF page-by-page via NuExtract 3."""
    import torch
    from PIL import Image

    pages, tmp_in = rasterise_pdf(pdf_path, dpi=dpi)
    template_str = json.dumps(TEMPLATE, indent=2)

    page_texts: list[str] = []
    page_raws: list[str] = []
    t0 = time.time()

    # Cap the longest edge so a 300-DPI tabloid page (3000+px) doesn't OOM
    # the T4 vision encoder. Qwen3.5-VL handles variable input sizes; the
    # processor will further bucket by patch size. 1280 is a reasonable
    # compromise between detail retention and VRAM headroom (~10 GB model
    # + activation slack on a 16 GB T4).
    MAX_EDGE = 1280

    for idx, page_path in enumerate(pages, 1):
        img = Image.open(page_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if max(img.size) > MAX_EDGE:
            ratio = MAX_EDGE / max(img.size)
            img = img.resize(
                (int(img.size[0] * ratio), int(img.size[1] * ratio)),
                Image.LANCZOS,
            )

        messages = [{"role": "user", "content": [{"type": "image", "image": img}]}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            template=template_str,
            enable_thinking=False,
        ).to(model.device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )

        trimmed = generated[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        page_raws.append(response)
        page_texts.append(parse_full_text(response))
        print(f"    page {idx}/{len(pages)} done", flush=True)

        # Free per-page activations so the next page doesn't OOM on T4
        del inputs, generated, trimmed
        torch.cuda.empty_cache()

    elapsed = time.time() - t0

    text = "\n\n".join(
        f"--- page {i + 1} ---\n\n{t}" for i, t in enumerate(page_texts)
    )

    # Audit trail — keep raw responses so we can see what NuExtract emitted
    raw_keep = OUTPUTS_DIR / "_raw" / pdf_path.stem
    raw_keep.mkdir(parents=True, exist_ok=True)
    (raw_keep / "nuextract3_raw.json").write_text(
        json.dumps({"pages": page_raws, "template": TEMPLATE}, indent=2)
    )

    # Clean up rasterised pages
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
        "template": TEMPLATE,
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
    (out_dir / f"{stem}.nuextract3.txt").write_text(text)
    (out_dir / f"{stem}.nuextract3.meta.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__)
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

    print(f"Found {len(pdfs)} PDF(s) under docs/.", flush=True)

    processor, model, dtype_label = load_model()

    summary: list[dict] = []
    for i, (src, pdf) in enumerate(pdfs, 1):
        out_dir = OUTPUTS_DIR / src
        out_txt = out_dir / f"{pdf.stem}.nuextract3.txt"
        if out_txt.exists() and not args.force:
            existing = json.loads(
                (out_dir / f"{pdf.stem}.nuextract3.meta.json").read_text()
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
        meta = run_one_pdf(pdf, processor, model, dpi=args.dpi)
        meta["dtype"] = dtype_label
        print(
            f"    done in {meta['elapsed_s']}s — {meta['n_pages']} page(s), "
            f"{meta['char_count']} chars",
            flush=True,
        )
        write_outputs(out_dir, pdf.stem, dict(meta))
        summary.append(
            {"src": src, "stem": pdf.stem, **{k: v for k, v in meta.items() if k != "text"}}
        )

    (OUTPUTS_DIR / "summary_nuextract3.json").write_text(json.dumps(summary, indent=2))
    print("\nDone. Summary written to outputs/summary_nuextract3.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
