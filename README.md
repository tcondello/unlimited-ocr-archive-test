# unlimited-ocr-archive-test

> Putting Baidu's [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) — a 3B vision-language model released this month — against the OCR engines I've been using on real archival documents: a handwritten 1966 Vietnam-era postcard, a typed 1968 letter, and two pages of a 1919 U.S. Army hospital newspaper.

Four documents, three engines, one repo. Clone, run on a GPU, eyeball the diffs.

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
compare.py                               # Builds outputs/comparison.md from baselines/ + outputs/

outputs/                                 # Populated by unlimited_ocr_test.py
  vva/*.unlimited-ocr.{txt,meta.json}
  asyouwere/*.unlimited-ocr.{txt,meta.json}
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

**Requirements:** NVIDIA GPU with BF16 support (Colab T4 is enough — the model is 3B), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:tcondello/unlimited-ocr-archive-test.git
cd unlimited-ocr-archive-test

cp .env.example .env
# Put your HF read token in .env so transformers can fetch the model

uv run unlimited_ocr_test.py          # all four docs, Gundam mode (default)
uv run unlimited_ocr_test.py --mode base   # alternative: base single-image mode
uv run compare.py                     # generate outputs/comparison.md
```

The script is per-item resumable — kill it and re-run, it picks up where it left off.

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

## What "Unlimited-OCR" is

A 3-billion-parameter vision-language model from Baidu, MIT licensed, released to Hugging Face in November 2026. The pitch is "one-shot long-horizon parsing" — meaning it ingests a whole page (or multiple pages in a single call) and emits up to 32k tokens of structured text. Two modes ship with the model:

- **Gundam** (`base_size=1024, image_size=640, crop_mode=True`) — cropping enabled. Better for dense or large pages.
- **Base** (`base_size=1024, image_size=1024, crop_mode=False`) — no cropping. Better for a single clean page at native resolution.

Both modes are run from the same script — `--mode gundam` (default) and `--mode base`.

## Results

After the first GPU run, `outputs/comparison.md` and `outputs/summary.json` will be committed alongside the script. I'll update this section with the headline numbers and a few honest observations once I've eyeballed the diffs myself — including where Unlimited-OCR shows up the existing baselines and where it doesn't.

**The interesting questions:**

1. Can a 3B open-weights VLM match Claude Sonnet on handwriting? That's the only place Claude has been clearly worth its $0.04-per-3-page cost on these archives.
2. Does it preserve column order on multi-column newsprint? Docling does because it has explicit layout analysis; pure VLMs often lose it.
3. What's the wall-clock per page? A T4 is ~$0.30/hr; if this is fast, it's structurally cheaper than per-call API pricing.

## License

[Apache 2.0](LICENSE). The model itself is MIT-licensed by Baidu. Document sources: VVA materials are public-domain U.S. Government and personal correspondence held by the Vietnam Veterans of America archive; ASYOUWERE is a 1919 U.S. Army publication, public domain.
