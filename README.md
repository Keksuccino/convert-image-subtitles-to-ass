# subtitle_convert (PGS -> ASS via OpenRouter)

This converts image-based PGS subtitles inside `.mkv` files into `.ass` subtitles by:

1. Extracting the PGS track(s) (`.sup`) from each MKV
2. Decoding each subtitle image
3. Sending **one OpenRouter request per subtitle image** using a vision-capable model
4. Writing an `.ass` file with the returned text

## Setup

System deps (Ubuntu/WSL):

```bash
sudo apt-get update
sudo apt-get install -y mkvtoolnix ffmpeg python3-venv
```

Python deps:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## OpenRouter API key

Create a `.env` file (or export env vars) with:

```bash
OPENROUTER_API_KEY=YOUR_KEY_HERE
```

Optional:

```bash
OPENROUTER_SITE_URL=https://example.com
OPENROUTER_APP_NAME=subtitle_convert
OPENROUTER_MODEL=google/gemini-2.5-flash
```

## Run

Dry-run (no API calls, useful to validate extraction/parsing/output):

```bash
.venv/bin/python convert_pgs_mkv_to_ass_openrouter.py --dry-run --limit-images 2
```

Real run (uses `.env` / `OPENROUTER_API_KEY`):

```bash
.venv/bin/python convert_pgs_mkv_to_ass_openrouter.py
```

Notes:

- Source MKV files should be in a folder called `episodes`, e.g. `/episodes/cool_tv_show_S01E02.mkv`
- Progress output can be disabled with `--no-progress`.
- Default track selection is `--track-policy all` (process every PGS subtitle track found).
- To force a specific PGS track id (from `mkvmerge -J`), use `--track-id 4` (repeatable).
- Outputs go to `episodes/ass/<mkv-stem>/...*.ass`
- Cache goes to `.work/openrouter_cache.json` (re-runs skip already-OCR’d images).
