# unlimited-ocr-archive-test

> Putting Baidu's [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) — a 3B vision-language model released this month — against the engines I've been using on real archival documents: NuMind's [NuExtract 3](https://huggingface.co/numind/NuExtract3) (my default open-weights extractor), plus the API-grade baselines from prior runs (Claude Sonnet 4.5 vision for Vietnam-era handwritten material, docling for newsprint).

Four documents, three engines per doc, one repo. Clone, run on a GPU, eyeball the diffs.

## What's in here

```
docs/
  vva/                                   # Vietnam Veterans of America archive
    2710101001.pdf                       # Postcard, July 1966 — handwritten + photo (2 pages)
    23930102001.pdf                      # Letter to "Mom," March 1968 — typewriter + handwriting (3 pages)
  asyouwere/                             # ASYOUWERE — U.S. Army General Hospital No. 24 newspaper
    52420710RX1.pdf                      # Vol. I No. 1, Feb 15 1919 — printed broadsheet
    52420710RX2.pdf                      # Vol. I No. 2, Feb 22 1919 — printed broadsheet

baselines/                               # OCR I already ran with other engines
  vva/*.claude-sonnet-4-5.txt            # Claude vision (sonnet 4.5) — with .meta.json (elapsed, cost, tokens)
  asyouwere/*.docling.{txt,md}           # docling (the IBM document conversion pipeline)

unlimited_ocr_test.py                    # Runs Unlimited-OCR over docs/, writes outputs/
nuextract3_test.py                       # Runs NuExtract 3 over docs/, writes outputs/
compare.py                               # Builds outputs/comparison.md from baselines/ + outputs/

outputs/                                 # Populated by the two test scripts
  vva/*.unlimited-ocr.{txt,meta.json}
  vva/*.nuextract3.{txt,meta.json}
  asyouwere/*.unlimited-ocr.{txt,meta.json}
  asyouwere/*.nuextract3.{txt,meta.json}
  _raw/                                  # Audit trail — raw model output before parsing
  comparison.md                          # Side-by-side table + excerpts
  summary.json                           # All metadata in one place
```

## Why these four documents

The point isn't to score one engine as "best." It's to see where each one breaks. The set deliberately spans the failure modes that matter for historical-archive OCR:

| Doc | Era | Medium | Hard parts |
|---|---|---|---|
| `2710101001.pdf` | 1966 | Handwritten ballpoint on postcard, faded | Cursive, low contrast, marginalia |
| `23930102001.pdf` | 1968 | Typewriter + handwritten signatures | Ink bleed, irregular spacing, multi-page continuity |
| `52420710RX1.pdf` | 1919 | Newsprint, multi-column | Column order, fraktur-adjacent typeface, age artifacts |
| `52420710RX2.pdf` | 1919 | Newsprint, multi-column | Same, plus a half-page photo and a table |

The Vietnam-era docs come from a [larger archive-OCR pipeline](https://github.com/tcondello) I've been running for retrieval; the 1919 newspapers come from a separate project doing the same exercise on a different corpus. Both already had OCR ground truth from prior engines — Claude vision for the Vietnam docs (the only thing that handled the handwriting well), docling for the newspapers (the multi-column layout broke the simpler engines).

## Run it

**Requirements:** NVIDIA GPU (Colab T4 is enough — Unlimited-OCR is 3B BF16, NuExtract 3 is 4B and runs in fp16 on T4), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:tcondello/unlimited-ocr-archive-test.git
cd unlimited-ocr-archive-test

cp .env.example .env
# Put your HF read token in .env so transformers can fetch the models

uv run unlimited_ocr_test.py          # all four docs, Gundam mode (default)
uv run nuextract3_test.py             # same four docs, NuExtract 3 with {"full_text": "verbatim-string"}
uv run compare.py                     # generate outputs/comparison.md
```

Both scripts are per-item resumable — kill them and re-run, they pick up where they left off.

### Running on Colab without a local GPU

The script is a [PEP 723](https://peps.python.org/pep-0723/) single file. To run it on a managed Colab GPU using the [Colab CLI](https://github.com/googlecolab/google-colab-cli):

```bash
# Easiest: clone this repo on the Colab VM
colab run --gpu T4 -- bash -c "
  git clone https://github.com/tcondello/unlimited-ocr-archive-test &&
  cd unlimited-ocr-archive-test &&
  uv run unlimited_ocr_test.py
"
```

For an opinionated `colab-hf-run` wrapper that handles HF auth and session lifecycle, see [`tcondello/uv-scripts-colab`](https://github.com/tcondello/uv-scripts-colab).

## The two open-weights engines

**Unlimited-OCR.** 3 B-parameter vision-language model from Baidu, MIT-licensed, released to Hugging Face this month. The pitch is "one-shot long-horizon parsing" — it ingests a whole page (or multiple pages in a single call) and emits up to 32 k tokens of structured text. Two modes ship with the model:

- **Gundam** (`base_size=1024, image_size=640, crop_mode=True`) — cropping enabled. Better for dense or large pages.
- **Base** (`base_size=1024, image_size=1024, crop_mode=False`) — no cropping. Better for a single clean page at native resolution.

Run with `--mode gundam` (default) or `--mode base`.

**NuExtract 3.** NuMind's 4 B vision-language *structured extractor*, built on Qwen3.5-VL. Strictly it's not an OCR engine — you give it a JSON template and it returns structured fields. For an OCR baseline I'm calling it with the simplest possible template:

```json
{"full_text": "verbatim-string"}
```

The model returns `{"full_text": "..."}` and the script pulls that string out as the document's text. This is a slightly off-label use — NuExtract's real superpower is "extract these specific fields from this document," which is how I use it in production — but for a pure-text comparison it's a fair lift on the same GPU.

## Results

After the first GPU run, `outputs/comparison.md` and `outputs/summary.json` will be committed alongside the script. I'll update this section with the headline numbers and a few honest observations once I've eyeballed the diffs myself — including where Unlimited-OCR shows up the existing baselines and where it doesn't.

**The interesting questions:**

1. Can either 3–4 B open-weights VLM match Claude Sonnet on handwriting? That's the only place the API engine has been clearly worth its ~$0.04-per-3-page cost on these archives.
2. Does either preserve column order on multi-column newsprint? Docling does because it has explicit layout analysis; pure VLMs often lose it.
3. Off-label vs. on-label: how does NuExtract 3 do when you ask it for verbatim text vs. its intended structured-fields use? If it's close, you can keep one model serving two jobs.
4. Wall-clock per page on the same T4 — which one is fastest, and is either structurally cheaper than per-call API pricing?

## License

[Apache 2.0](LICENSE). Models: Unlimited-OCR is MIT (Baidu); NuExtract 3 is MIT (NuMind). Document sources: VVA materials are public-domain U.S. Government and personal correspondence held by the Vietnam Veterans of America archive; ASYOUWERE is a 1919 U.S. Army publication, public domain.
