# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface-hub>=0.25",
# ]
# ///
"""scripts/pull_results.py — pull the bake-off outputs from HF back into ./outputs.

After running recipes/colab_runner.py on Colab (which uploads outputs/ as
an HF dataset), this script downloads that dataset and lays the files out
under the project's local outputs/ directory.

Env vars:
  HF_DATASET   HF dataset id to pull from (required)
  HF_TOKEN     Optional — for private datasets

Usage:
  HF_DATASET=tcondello/ocr-archive-test-results uv run scripts/pull_results.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def main() -> int:
    from huggingface_hub import snapshot_download

    dataset = os.environ.get("HF_DATASET", "").strip()
    if not dataset:
        sys.exit("ERROR: HF_DATASET env var is required (e.g. tcondello/ocr-results).")
    token = os.environ.get("HF_TOKEN") or None

    print(f"Pulling {dataset} (dataset) into a snapshot cache...", flush=True)
    local_path = snapshot_download(
        repo_id=dataset,
        repo_type="dataset",
        token=token,
    )
    print(f"  cached at {local_path}", flush=True)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in Path(local_path).rglob("*"):
        if src.is_dir() or src.name.startswith("."):
            continue
        rel = src.relative_to(local_path)
        # HF snapshot may include README.md / .gitattributes at the root —
        # skip those, only sync project-shaped paths.
        if rel.parts[0] in ("README.md", ".gitattributes"):
            continue
        dest = OUTPUTS_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    print(f"Copied {copied} file(s) into {OUTPUTS_DIR.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
