# Inkwell

Inkwell is a local OCR/annotation pipeline for handwritten notebook pages.

## Current Status

- Phase 0: DB schema + asset ingestion complete
- Phase 1: orientation/layout ingest detection + ingest review UI complete
- Phase 2 (active): annotation workflow implemented and in use
	- `/annotate` line correction UI
	- random next-line sampling for unannotated OCR lines
	- review/edit routes: `/annotate/review`, `/annotate/edit/<line_id>`
	- context crop endpoint: `/annotate/api/context/<line_id>`
	- shorthand markers: `[ur]`, `[nt]`, `[?]`
	- optional flags: `SEGMENTATION_ISSUE`, `UNUSABLE_SEGMENTATION`, `NOT_TEXT`
	- immutable human annotations

Current working direction:
- full-source preprocessing and segmentation are complete,
- full OCR coverage is expected/active after the latest full OCR run,
- annotation is ongoing,
- next engineering steps are split assignment, GT export, model comparison, and fine-tuning.

## Repo Structure

- `inkwell/` — Python package (db, config, pipeline, web)
- `scripts/` — operational scripts
- `working/` — runtime artifacts (DB, derived files)
- `docs/` — current implementation, roadmap, and handoff documentation

## Prerequisites

- Python (the environment where you run `python`)
- SQLite (bundled with Python)
- Tesseract binary for orientation OSD (`pytesseract` wrapper alone is not enough)

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Optional ML/training dependencies:

```bash
pip install -r requirements-ml.txt
```

Install Tesseract binary (Linux):

```bash
sudo apt-get install tesseract-ocr
```

## Initialize / Reinitialize the Database

Use the main config (8 notebooks):

```bash
python scripts/init_db.py --config notebooks_config.json
```

Expected behavior:
- Idempotent reruns (no duplicate assets)
- DB created at `working/inkwell.db`

## Run Phase 1 Ingest Detection

```bash
python scripts/run_pipeline.py ingest
```

What it does:
- orientation detection (Tesseract OSD)
- layout detection (`DOUBLE` if width > 1.5 * height, otherwise `SINGLE`)
- saves results to `source_images.orientation_detected`, `source_images.layout_type`

If Tesseract is missing, orientation defaults to `0` and warnings are printed.

## Run Ingest Review Web UI

Start server:

```bash
python scripts/run_web.py --debug
```

Open:
- `http://127.0.0.1:5000/ingest`
- `http://127.0.0.1:5000/annotate/`

If port 5000 is already used:

```bash
python scripts/run_web.py --debug --port 5001
```

## Annotation Notes

- Progress shows `annotated/total (percent)` where total is distinct OCR lines.
- Human annotation writes immutable rows (`immutable=1`).
- Queue excludes lines already marked `HUMAN_CORRECTED` or `FLAGGED`.

## Full Pipeline Commands

Initialize DB and assets:

```bash
python scripts/init_db.py --config notebooks_config.json
```

Run ingest detection:

```bash
python scripts/run_pipeline.py ingest
```

Preprocess full source (rotation, deskew, split double pages):

```bash
python scripts/run_pipeline.py preprocess --force
```

Segment full source into lines:

```bash
python scripts/run_pipeline.py segment --force
```

Run OCR on all segmented lines:

```bash
python scripts/run_pipeline.py ocr --model easyocr --langs hu,en
```

## Live Docs To Read First

For a new session, start with these compact active docs:

- `docs/inkwell_plan_final.md` — canonical current plan and roadmap
- `docs/runbook.md` — exact operational commands and loops
- `docs/session_handoff.md` — concise current state + next priorities

Historical / superseded material is kept under:

- `docs/archive/`

