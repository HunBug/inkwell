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

## 2026-03-10 — Multi-model OCR testing (EasyOCR + TrOCR) + visual comparison

### What changed
- OCR pipeline supports multiple model backends via `--model`
- Implemented per-model resume/skip logic (`created_by`) so different engines can run on the same lines
- Added visual sampler filter by model (`scripts/sample_ocr.py --model ...`)

### How to test
1. Activate project venv:
	```bash
	source .venv/bin/activate
	```
2. Install OCR model dependencies (if not already installed):
	```bash
	pip install -r requirements.txt
	pip install -r requirements-ml.txt
	pip install transformers sentencepiece
	```
3. Run EasyOCR sample batch:
	```bash
	python scripts/run_pipeline.py ocr --model easyocr --limit 50
	```
4. Run TrOCR sample batch (first run downloads model weights, can take time):
	```bash
	python scripts/run_pipeline.py ocr --model trocr --limit 50
	```
5. Generate model-specific visual sample pages:
	```bash
	python scripts/sample_ocr.py --model easyocr --size 30 --output working/ocr_sample_easyocr.html
	python scripts/sample_ocr.py --model trocr --size 30 --output working/ocr_sample_trocr.html
	```
6. Open results in browser:
	- `working/ocr_sample_easyocr.html`
	- `working/ocr_sample_trocr.html`
7. Compare DB counts per model:
	```bash
	sqlite3 working/inkwell.db "SELECT created_by, COUNT(*) FROM transcriptions WHERE transcription_type='OCR_AUTO' GROUP BY created_by ORDER BY created_by;"
	```
8. Compare latest outputs side-by-side in terminal:
	```bash
	sqlite3 working/inkwell.db "SELECT created_by, model_version, line_id, substr(text,1,120) FROM transcriptions WHERE transcription_type='OCR_AUTO' ORDER BY id DESC LIMIT 30;"
	```

### Expected result
- EasyOCR and TrOCR both run without schema conflicts
- `transcriptions` contains rows for both `created_by='easyocr'` and `created_by='trocr'`
- Visual samples render line image + OCR text for each model
- Re-running one model without `--force` skips lines already processed by that same model only

### Actual result / notes
- EasyOCR run completed and wrote rows under `created_by='easyocr'`
- TrOCR pipeline is wired and executable; first run may spend significant time downloading checkpoint weights
- If TrOCR is interrupted during first download, rerun the same command to continue once cache is populated

## 2026-03-10 — Getting Hungarian TrOCR checkpoint (OCR_HU_Tra2022) and testing in Inkwell

### What changed
- Added support for custom TrOCR checkpoint IDs via `--model "trocr:<checkpoint_id>"`
- Added practical steps to acquire Hungarian checkpoint used by OCR_HU_Tra2022
- Added test commands to compare default TrOCR vs Hungarian checkpoint

### How to test
1. Activate environment and install runtime deps:
	```bash
	source .venv/bin/activate
	pip install -r requirements.txt
	pip install -r requirements-ml.txt
	pip install transformers sentencepiece huggingface_hub
	```

2. Login to Hugging Face (needed if model is gated/private):
	```bash
	huggingface-cli login
	```

3. Checkpoint from OCR_HU_Tra2022 inference script:
	- Repo inference default points to:
	  - `AlhitawiMohammed22/trocr_large_lines_v2_1_ft_on_dh-lab`

4. Run baseline TrOCR (English-leaning):
	```bash
	python scripts/run_pipeline.py ocr --model trocr --limit 50
	```

5. Run Hungarian checkpoint candidate:
	```bash
	python scripts/run_pipeline.py ocr --model "trocr:AlhitawiMohammed22/trocr_large_lines_v2_1_ft_on_dh-lab" --limit 50
	```

6. If checkpoint access fails (404/403):
	- request access from model owner profile (`AlhitawiMohammed22`), or
	- contact listed repo author emails in OCR_HU_Tra2022 README.

7. Generate visual samples per model:
	```bash
	python scripts/sample_ocr.py --model "trocr:microsoft/trocr-base-handwritten" --size 30 --output working/ocr_sample_trocr_base.html
	python scripts/sample_ocr.py --model "trocr:AlhitawiMohammed22/trocr_large_lines_v2_1_ft_on_dh-lab" --size 30 --output working/ocr_sample_trocr_hu.html
	```

8. Compare model counts and latest rows:
	```bash
	sqlite3 working/inkwell.db "SELECT created_by, COUNT(*) FROM transcriptions WHERE transcription_type='OCR_AUTO' GROUP BY created_by ORDER BY created_by;"
	sqlite3 working/inkwell.db "SELECT created_by, model_version, line_id, substr(text,1,120) FROM transcriptions WHERE transcription_type='OCR_AUTO' ORDER BY id DESC LIMIT 40;"
	```

### Expected result
- Hungarian checkpoint is accepted by Inkwell through `trocr:<checkpoint>` syntax
- OCR rows are stored separately per model in `created_by`
- Visual sample for Hungarian checkpoint shows more Hungarian-like lexical output than default TrOCR baseline

### Actual result / notes
- Inference script in OCR_HU_Tra2022 references `AlhitawiMohammed22/trocr_large_lines_v2_1_ft_on_dh-lab`
- Some project assets/models may be private or moved; authenticated HF access may be required
- Inkwell now supports direct checkpoint swap without code changes
