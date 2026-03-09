# Inkwell Testing Guide

This document is for manual testers and is **append-only** for change history.

## 1) Test Environment Initialization

### 1.1 Install dependencies

```bash
pip install -r requirements.txt
```

Optional (only for ML training/inference stack):

```bash
pip install -r requirements-ml.txt
```

### 1.2 Install system OCR binary (required for orientation detection)

```bash
sudo apt-get install tesseract-ocr
```

### 1.3 Initialize database from configured notebooks

```bash
python scripts/init_db.py --config notebooks_config.json
```

Expected output (example):
- `Init complete`
- `Notebooks: 8`
- `Assets: 498 (...)`
- `Pages: 498`

### 1.4 Run ingest detection

```bash
python scripts/run_pipeline.py ingest
```

Expected:
- prints ingest summary
- updates `source_images.orientation_detected`
- updates `source_images.layout_type`

### 1.5 Start web UI for ingest review

```bash
python scripts/run_web.py --debug
```

Open:
- `http://127.0.0.1:5000/ingest`

If port busy:

```bash
python scripts/run_web.py --debug --port 5001
```

## 2) What To Test (Current Phase)

### 2.1 Ingest CLI

- command runs without crash
- summary shows processed image count
- if Tesseract missing: warnings appear, pipeline still finishes

### 2.2 Ingest UI

- `/ingest` page loads
- image cards are shown with filename/notebook/status
- orientation/layout dropdowns are editable
- single `Confirm` updates status to confirmed
- bulk selection + `Bulk Confirm Selected` works
- pagination works between pages

## 3) Regression Smoke Test (Quick)

1. `python scripts/init_db.py --config notebooks_config.json`
2. `python scripts/run_pipeline.py ingest`
3. `python scripts/run_web.py --debug --port 5001`
4. Open `/ingest`
5. Confirm one item and one bulk set
6. Reload page and verify confirmed status persists

## 4) Change Log (Append New Entries Only)

Use this template for each new feature/fix:

```md
## YYYY-MM-DD — <feature or fix>

### What changed
- ...

### How to test
1. ...
2. ...

### Expected result
- ...

### Actual result / notes
- ...
```

---

## 2026-03-09 — Phase 1 ingest detection + ingest review UI

### What changed
- Added `scripts/run_pipeline.py ingest` implementation
- Added orientation detection via Tesseract OSD (`pytesseract`)
- Added layout detection heuristic (`DOUBLE` if width > 1.5 * height)
- Added Flask app + ingest route (`/ingest`) with manual confirmation
- Added single confirm and bulk confirm endpoints

### How to test
1. Run `python scripts/init_db.py --config notebooks_config.json`
2. Run `python scripts/run_pipeline.py ingest`
3. Run `python scripts/run_web.py --debug --port 5001`
4. Open `http://127.0.0.1:5001/ingest`
5. Confirm at least 1 item individually and 3+ items in bulk

### Expected result
- Ingest command completes and prints processed count
- Ingest UI loads cards and allows confirmation updates
- Confirmed state remains after refresh

### Actual result / notes
- Ingest processed all 498 images successfully
- Layout detection completed
- Orientation fell back to `0°` when Tesseract binary not installed
- Web app startup/import issues were fixed (path bootstrapping + circular import)

## 2026-03-09 — Requirements compatibility update (torch/kraken conflict)

### What changed
- Updated `requirements.txt` with modern version ranges for core dependencies
- Split heavy/optional ML dependencies into `requirements-ml.txt`
- Added Python marker for `torch` (`python_version < 3.14`) in `requirements-ml.txt`
- Kept `kraken` guarded behind `python_version < 3.11` in `requirements-ml.txt` because upstream `kraken` requires old `torch<1.11`

### How to test
1. Run `pip install -r requirements.txt`
2. Verify install completes without resolver conflict
3. (Optional) run `pip install -r requirements-ml.txt`
4. Run `python scripts/init_db.py --config notebooks_config.json`
5. Run `python scripts/run_pipeline.py ingest`

### Expected result
- `pip install -r requirements.txt` succeeds on modern Python (no `ResolutionImpossible`)
- `requirements-ml.txt` installs only where Python/version markers allow
- Ingest pipeline still runs

### Actual result / notes
- Compatibility strategy implemented to prevent `torch`/`kraken` resolver conflict
- If full Kraken segmentation is needed later, use a dedicated older Python environment (recommended: 3.10)
