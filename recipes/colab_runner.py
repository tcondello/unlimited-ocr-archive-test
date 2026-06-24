# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface-hub>=0.25",
#   "rich>=13.0",
# ]
# ///
"""recipes/colab_runner.py — run the OCR bake-off on a managed Colab GPU.

Designed for tcondello/uv-scripts-colab's `bin/colab-hf-run` wrapper, which
forwards `HF_TOKEN` + `OUTPUT_DATASET` into the Colab kernel.

On the Colab VM, the recipe:
  1. git clones github.com/tcondello/unlimited-ocr-archive-test (or pulls)
  2. pip-installs the OCR scripts' deps (torch, transformers, pymupdf, ...)
  3. runs unlimited_ocr_test.py as a subprocess  (VRAM freed on exit)
  4. runs nuextract3_test.py as a subprocess     (fresh slate for the 4B model)
  5. runs compare.py to build outputs/comparison.md
  6. uploads outputs/ to OUTPUT_DATASET as a dataset repo via
     huggingface_hub.HfApi().upload_folder(...)

Pull the results locally afterwards with:
  HF_DATASET=$OUTPUT_DATASET uv run scripts/pull_results.py

Env vars:
  OUTPUT_DATASET   HF dataset id to push outputs/ to (required)
  HF_TOKEN         Required — auto-injected by colab-hf-run from your local cache
  REPO_URL         Override repo to clone               [tcondello/unlimited-ocr-archive-test]
  REPO_REF         Branch / tag / commit               [main]
  ENGINES          Comma-separated: "unlimited,nuextract" [unlimited,nuextract]

Usage:
  OUTPUT_DATASET=tcondello/ocr-archive-test-results \\
    bin/colab-hf-run recipes/colab_runner.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

DEFAULT_REPO_URL = "https://github.com/tcondello/unlimited-ocr-archive-test"
DEFAULT_REPO_REF = "main"
WORK_DIR = "/content/unlimited-ocr-archive-test"

OCR_DEPS = [
    "torch>=2.4",
    "torchvision",
    "transformers>=4.49",
    "accelerate>=0.30",
    "Pillow",
    "pymupdf",
    "python-dotenv",
    # Unlimited-OCR's custom modeling code (trust_remote_code=True) imports these
    # directly; they're listed in the model-card requirements.txt.
    "addict",
    "easydict",
    "einops",
    "matplotlib",
    "psutil",
]


def _run(cmd: list[str], cwd: str | None = None, check: bool = True) -> int:
    """Run a command, streaming stdout/stderr line-by-line, returning the exit code.

    Colab's IPython kernel doesn't reliably surface a subprocess's inherited
    stdout/stderr, so we pipe explicitly and forward each line. This keeps
    OCR-script progress and tracebacks visible in the colab-hf-run stream.
    """
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if check and proc.returncode != 0:
        raise SystemExit(
            f"command failed (exit {proc.returncode}): {' '.join(cmd)}"
        )
    return proc.returncode


def _ensure_repo(repo_url: str, repo_ref: str, work_dir: str) -> None:
    if os.path.isdir(os.path.join(work_dir, ".git")):
        print(f"[setup] repo already cloned at {work_dir} — pulling...", flush=True)
        _run(["git", "-C", work_dir, "fetch", "origin"])
        _run(["git", "-C", work_dir, "checkout", repo_ref])
        _run(["git", "-C", work_dir, "pull", "--ff-only", "origin", repo_ref])
        return
    print(f"[setup] cloning {repo_url}@{repo_ref} → {work_dir}", flush=True)
    _run(["git", "clone", "--depth", "1", "--branch", repo_ref, repo_url, work_dir])


def _ensure_deps() -> None:
    """pip-install the OCR scripts' dependencies into the Colab kernel."""
    # Probe via subprocess so we don't pin stale modules into sys.modules
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, transformers, accelerate, fitz, PIL, dotenv, "
            "addict, easydict, einops, matplotlib, psutil",
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        print("[setup] OCR deps already present.", flush=True)
        return

    print("[setup] installing OCR deps...", flush=True)
    _run([sys.executable, "-m", "pip", "install", "-q", *OCR_DEPS])


def _run_engine(work_dir: str, script: str) -> None:
    print(f"\n[engine] running {script}", flush=True)
    t0 = time.time()
    _run([sys.executable, script], cwd=work_dir)
    print(f"[engine] {script} finished in {time.time() - t0:.1f}s", flush=True)


def _upload_outputs(work_dir: str, output_dataset: str, hf_token: str) -> str:
    from huggingface_hub import HfApi

    outputs_dir = os.path.join(work_dir, "outputs")
    if not os.path.isdir(outputs_dir):
        raise SystemExit(f"outputs/ directory missing under {work_dir}")

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=output_dataset, repo_type="dataset", exist_ok=True)
    print(f"\n[upload] pushing {outputs_dir} → {output_dataset} (dataset)", flush=True)
    api.upload_folder(
        folder_path=outputs_dir,
        repo_id=output_dataset,
        repo_type="dataset",
        commit_message="add ocr-bakeoff outputs (unlimited-ocr + nuextract3 + comparison)",
        ignore_patterns=[".gitkeep"],
    )
    return f"https://huggingface.co/datasets/{output_dataset}"


def main() -> int:
    output_dataset = os.environ.get("OUTPUT_DATASET", "").strip()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    repo_url = os.environ.get("REPO_URL", DEFAULT_REPO_URL).strip()
    repo_ref = os.environ.get("REPO_REF", DEFAULT_REPO_REF).strip()
    engines_raw = os.environ.get("ENGINES", "unlimited,nuextract").strip()
    engines = [e.strip() for e in engines_raw.split(",") if e.strip()]

    if not output_dataset:
        sys.exit("ERROR: OUTPUT_DATASET env var is required (e.g. tcondello/ocr-results).")
    if not hf_token:
        sys.exit(
            "ERROR: HF_TOKEN is required. colab-hf-run should inject it from "
            "your local ~/.cache/huggingface/token — did you run `hf auth login`?"
        )

    print(
        f"[config] repo={repo_url}@{repo_ref}\n"
        f"[config] engines={engines}\n"
        f"[config] output={output_dataset}",
        flush=True,
    )

    _ensure_repo(repo_url, repo_ref, WORK_DIR)
    _ensure_deps()

    if "unlimited" in engines:
        _run_engine(WORK_DIR, "unlimited_ocr_test.py")
    if "nuextract" in engines:
        _run_engine(WORK_DIR, "nuextract3_test.py")

    print("\n[compare] building outputs/comparison.md", flush=True)
    _run([sys.executable, "compare.py"], cwd=WORK_DIR)

    url = _upload_outputs(WORK_DIR, output_dataset, hf_token)
    print(f"\nDone. Outputs at {url}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
