# Inkwell — Implementation Plan

> Read `inkwell_design.md` first. This document assumes full familiarity with the design.  
> This plan is ordered for a single developer. Each phase produces something runnable before the next phase starts.

---

## Phase 0 — Repository and Environment Setup

**Goal:** Empty but runnable project. DB schema exists. Config can be read.

### Tasks

1. Create repo `inkwell`, initialise git, create `.gitignore` (exclude `working/`, `*.db`, `__pycache__`)
2. Create `venv`, generate `requirements.txt`

    ```
    # Core pipeline
    torch
    transformers
    datasets
    kraken
    opencv-python
    Pillow
    
    # Web UI
    flask
    
    # Utilities
    numpy
    scikit-learn
    tqdm
    ```

3. Create package structure as defined in the design document (all files empty for now)
4. Implement `inkwell/db.py`:
    - `get_connection()` — returns SQLite connection with `row_factory = sqlite3.Row`
    - `create_schema()` — runs all `CREATE TABLE IF NOT EXISTS` statements from the data model
    - Include SQLite triggers enforcing immutability:
        ```sql
        CREATE TRIGGER protect_gt_transcription
        BEFORE UPDATE ON transcriptions
        WHEN OLD.immutable = 1
        BEGIN SELECT RAISE(ABORT, 'Cannot modify immutable transcription'); END;
        
        CREATE TRIGGER protect_gt_segmentation
        BEFORE UPDATE ON segmentations
        WHEN OLD.immutable = 1
        BEGIN SELECT RAISE(ABORT, 'Cannot modify immutable segmentation'); END;
        ```
5. Implement `inkwell/config.py`:
    - `get_root_path(override=None)` — returns override if provided, else reads from config table, raises clear error if not set
    - `resolve_path(relative_path)` — joins root_path with relative path
    - `set_root_path(path)` — writes to config table
6. Implement `scripts/init_db.py`:
    - Creates schema
    - Reads `notebooks_config.json`
    - Populates `notebooks`, `handwriting_profiles` tables
    - Sets `root_path` in config table
    - Scans each notebook folder, creates `assets` rows ordered by filename counter
    - Idempotent: safe to run twice
7. Implement `scripts/set_root.py`:
    - One-liner: updates root_path in config table

### Validation

- `python scripts/init_db.py --config notebooks_config.json` runs without error
- `sqlite3 working/inkwell.db ".tables"` shows all expected tables
- `sqlite3 working/inkwell.db "SELECT filename FROM assets LIMIT 10"` shows correct files in order

---

## Phase 1 — Ingestion Pipeline (CLI)

**Goal:** All 500 source files are processed, metadata populated, working images exist.

### Tasks

1. Implement `inkwell/pipeline/ingest.py`:
    - For each `assets` row with `should_ocr=1` and no existing `source_images` row:
        - Run Tesseract OSD for orientation detection (`pytesseract.image_to_osd`)
        - Detect layout type: if image width > 1.5 × height → DOUBLE, else SINGLE
        - Create `source_images` row with detected values, `orientation_confirmed = NULL`
    - Log summary: N files processed, N detected as double-page, N detected as rotated

2. Implement `inkwell/pipeline/preprocess.py`:
    - For each `source_images` row where `orientation_confirmed` is set and `derived_image_path` is NULL:
        - Rotate image to `orientation_confirmed`
        - Deskew (OpenCV: find text angle, rotate to correct)
        - Binarise (adaptive threshold, configurable block size and constant)
        - Save to `working/derived_images/<asset_id>.jpg`
        - Update `source_images.derived_image_path`
    - Deskew algorithm: Hough line transform or minAreaRect on thresholded text blobs — use whichever is more stable, test on a sample of pages
    - Binarisation: try adaptive Gaussian threshold first; if bleed-through is visible, document that `unpaper` is the escalation path

3. Note: preprocess runs after ingestion review confirms orientation. It is not called automatically from ingest.

### Validation

- Run ingest on all 500 files
- Spot-check orientation detection: look at the summary log, manually review flagged pages
- After a few manual confirmations in Phase 2 UI, run preprocess and verify derived images look clean

---

## Phase 2 — Ingestion Review UI

**Goal:** Human can confirm orientation, layout, flag attachments, and draw skip regions. After this phase, all pages are ready for segmentation.

### Tasks

1. Implement `inkwell/web/app.py` — Flask app skeleton, register blueprints
2. Implement `inkwell/web/routes/ingest.py`:
    - `GET /ingest` — paginated grid of source_images, sorted by notebook then file_order
    - Each card shows: thumbnail, detected orientation, detected layout, asset_type badge
    - `POST /ingest/<source_image_id>/confirm` — update orientation_confirmed, layout_type, asset_type
    - Keyboard shortcuts: arrow keys to navigate, number keys for orientation (0/1/2/3), s to mark as skip
    - Bulk confirm: "mark all on this page as correct" button
3. Implement `inkwell/web/routes/regions.py`:
    - `GET /regions/<page_id>` — full page image with existing regions overlaid as SVG polygons
    - `POST /regions/<page_id>/add` — save new region (polygon_coords, region_type, behavior, flagged_by='ANNOTATOR')
    - `DELETE /regions/<region_id>` — delete region
    - Frontend: click to add polygon points, double-click to close, colour-coded by region_type
    - This view is linked from the ingest review card for each page

### Notes

- At this stage, `pages` rows do not yet exist — they are created by segmentation. The region editor can work against `source_images` for now, or you create stub `pages` rows during ingest. Decide which is cleaner during implementation. Creating stub pages rows during ingest is probably simpler.
- Keep the UI fast. The goal is to get through 500 files quickly. Keyboard-first design.

### Validation

- Navigate all 500 thumbnails, confirm ~10–20 representative pages
- Draw a few skip regions, verify they are stored correctly in DB
- Run preprocess on confirmed pages, spot-check output images

---

## Phase 3 — Segmentation

**Goal:** All diary pages have line polygons. Line crop images exist on disk.

### Tasks

1. Implement `inkwell/pipeline/segment.py`:
    - For each page where `processing_status = 'ingested'` (or `force_reprocess=1`):
        - Check for existing GT/HUMAN_CORRECTED segmentation — if found, use that instead of running Kraken
        - Run Kraken BaselineDet on `derived_image_path`
        - Store result as `segmentations` row (type=AUTO)
        - For each detected line:
            - Check if line polygon intersects any `SKIP_OCR` region — if so, `lines.skip=1`
            - Add vertical padding to crop bounding box (configurable, default 15% of line height)
            - Crop line image, save to `working/line_crops/<page_id>_<line_order>.jpg`
            - Create `lines` row
        - Update `pages.processing_status = 'segmented'`
    - Kraken API usage:
        ```python
        from kraken import blla
        from kraken.lib import vgsl
        model = vgsl.TorchVGSLModel.load_model(kraken.locate_model('blla.mlmodel'))
        result = blla.segment(image, model=model)
        # result.lines contains BaselineLine objects with .baseline and .boundary
        ```

2. Implement `inkwell/web/routes/segmentation.py` (optional review step):
    - `GET /segmentation/<page_id>` — page image with line polygons overlaid
    - Each line shown as a coloured quad
    - Click line to select, drag corners to adjust (simple quad editor)
    - Add line: click and drag to draw rectangle
    - Delete line: select + delete key
    - `POST /segmentation/<page_id>/approve` — saves corrected polygons as GT segmentation (immutable=1), regenerates line crops for changed lines

### Validation

- Run segmentation on 20–30 representative pages from different notebooks
- Open segmentation review for these pages, assess quality
- Test that a page with a skip region correctly marks those lines as skipped
- Verify line crop images look correct (line of text centred, padding visible)

---

## Phase 4 — Initial OCR and Fine-tuning Infrastructure

**Goal:** OCR pipeline runs. Fine-tuning pipeline runs. First model checkpoint produced.

### Tasks

1. Implement `inkwell/pipeline/ocr.py`:
    - Load TrOCR model (configurable checkpoint path, default `microsoft/trocr-base-handwritten`)
    - For each page in `processing_status = 'segmented'`:
        - For each non-skipped line on that page:
            - Run inference on `crop_image_path`
            - Extract text and per-token log-probabilities
            - Compute line confidence: mean of token log-probs
            - Store `transcriptions` row (type=OCR_RAW, confidence=line_confidence)
        - Update `pages.processing_status = 'ocr_complete'`
    - Batch inference for efficiency (batch size configurable, default 8)

2. Implement `inkwell/pipeline/finetune.py`:
    - Load all lines with GT or HUMAN_CORRECTED transcriptions in train split
    - Build HuggingFace Dataset from (crop_image_path, text) pairs
    - Fine-tune TrOCR using `Trainer` API
    - Save checkpoint to `working/checkpoints/<timestamp>/`
    - Log: training loss, validation CER (Character Error Rate) on val split
    - Model version string: `trocr-base-<timestamp>` stored in checkpoint metadata

3. HuggingFace Dataset class for this project:
    ```python
    class DiaryDataset(Dataset):
        def __init__(self, db_path, split, processor):
            # Load (image_path, text) pairs from DB for given split
            # processor = TrOCRProcessor.from_pretrained(...)
        def __getitem__(self, idx):
            # Load image, preprocess, tokenize text, return dict
    ```

4. Implement `scripts/run_pipeline.py` — CLI orchestrator:
    ```
    python scripts/run_pipeline.py ingest
    python scripts/run_pipeline.py preprocess
    python scripts/run_pipeline.py segment [--page <id>] [--force]
    python scripts/run_pipeline.py ocr [--model <path>] [--page <id>] [--force]
    python scripts/run_pipeline.py finetune [--model <path>]
    ```

### Validation

- Run OCR on 20 segmented pages, spot-check OCR_RAW transcriptions in DB
- Manually annotate 200 lines (using Phase 5 annotation UI once built, or directly in DB for now)
- Run finetune, verify checkpoint is saved and loss decreases
- Re-run OCR with new checkpoint on held-out pages, compare CER

---

## Phase 5 — Annotation UI (Active Learning Loop)

**Goal:** The main daily-use tool. Annotator works through lines efficiently. Model improves with each batch.

### Tasks

1. Implement `inkwell/pipeline/active_learn.py`:
    - `score_lines(session_id, model_version)`:
        - For all unlabelled lines (no GT/HUMAN_CORRECTED transcription, not skipped):
            - Uncertainty = 1 - OCR_RAW confidence
            - Diversity = distance to nearest already-labelled line embedding (use mean pixel histogram as embedding, cosine distance)
            - Priority = 0.7 × uncertainty + 0.3 × diversity
            - Insert into `active_learning_candidates`
    - `get_next_candidate(session_id)` — returns highest-priority unscored, unpresented candidate
    - `mark_annotated(candidate_id)` / `mark_skipped(candidate_id)`

2. Implement `inkwell/web/routes/annotate.py`:
    - `GET /annotate` — main annotation view
        - Fetch next candidate from active_learn
        - Display: line crop image (large, clear), text input pre-filled with OCR_RAW text
        - Show: confidence score, notebook name, approximate page date
        - Show: page thumbnail with current line highlighted (context)
    - `POST /annotate/submit` — save HUMAN_CORRECTED transcription (immutable=1), advance to next
    - `POST /annotate/skip` — mark candidate as skipped
    - `POST /annotate/gt` — save as GT (immutable=1), for lines annotator is very confident about
    - Keyboard: Tab = submit, Escape = skip, Ctrl+G = mark as GT
    - Progress indicator: X labelled / Y total, estimated lines until next fine-tune trigger
    - After every 100 submissions: show prompt "Run fine-tune now?" with one-click trigger

3. Hungarian spell-check integration (optional, implement after core flow works):
    - Use `hunspell` with Hungarian dictionary (`hu_HU`)
    - Python binding: `pyhunspell` or `spylls`
    - Highlight suspicious words in the text input
    - Click suspicious word: "Mark as intentional" button → sets `word_flags.verified_intentional=true`

### Validation

- Annotate 50 lines, verify all are stored correctly with immutable=1
- Skip 5 lines, verify they are marked skipped in candidates table
- Trigger a fine-tune run, verify new checkpoint is created
- Re-run OCR scoring on remaining candidates with new checkpoint, verify confidence improves on similar lines

---

## Phase 6 — Viewer

**Goal:** Read the diary. Navigate by notebook and page. Click any line to see source image.

### Notes

This is the lowest-priority phase. Implement after the annotation loop is working and producing good results. The database already contains everything needed — this phase is purely presentation.

### Tasks

1. Implement `inkwell/web/routes/viewer.py`:
    - `GET /view` — notebook list
    - `GET /view/<notebook_id>` — page list for notebook
    - `GET /view/<notebook_id>/<page_id>` — main view:
        - Left: transcribed text, one `<span>` per line, `[??]` and `[NT]` rendered with CSS classes
        - Right: page scan image with line highlight overlay
        - Click text span → highlight corresponding line region on image
        - Toggle: OCR_RAW / HUMAN_CORRECTED (future: LLM_POSTPROCESSED)
        - For `page_type != text`: image only, no text panel
        - ATTACHMENT pages: show image, any external_content text below
    - `GET /view/search?q=<query>` — full-text search over transcriptions, returns list of matching lines with page links

2. `inkwell/export.py`:
    - `export_text(notebook_id=None)` — dumps highest-authority transcription per line, ordered, to plain text
    - `export_json(notebook_id=None)` — full export with lineage: page_id, line_id, text, confidence, source image path

### Validation

- Navigate a complete notebook from start to finish
- Click several text spans, verify correct line is highlighted on image
- Search for a word you remember writing, verify it is found

---

## Recommended Implementation Order

```
Week 1:   Phase 0 (schema + config) + Phase 1 (ingest CLI)
Week 2:   Phase 2 (ingestion review UI) — get all 500 files confirmed
Week 3:   Phase 3 (segmentation) — get all pages segmented
Week 4:   Phase 4 (OCR + finetune infrastructure) — first model running
Week 5+:  Phase 5 (annotation UI) — this phase is open-ended, runs in parallel
          with iterative fine-tuning until accuracy is satisfactory
Later:    Phase 6 (viewer) — when you want to start reading the results
```

---

## Key Technical Decisions to Make During Implementation

**Deskew method:** Test both Hough line transform and minAreaRect approaches on a sample of your pages. The right choice depends on how much variation there is in page orientation. Document the chosen approach in `preprocess.py`.

**Binarisation parameters:** Adaptive threshold block size (typically 11–31 px) and constant (typically 2–10) need tuning for your scan DPI. Test on 10–20 pages before committing to defaults.

**Kraken padding:** Start with 15% vertical padding on line crops. Increase to 20–25% if accents are still being clipped. This is a configurable parameter in `segment.py`.

**Active learning trigger:** After how many annotations should fine-tuning be triggered? Suggested: first trigger at 200 labels, then every 100 after that. Configurable constant in `active_learn.py`.

**Confidence threshold for auto-accept:** If a line has OCR confidence above some threshold (e.g. 0.95), consider auto-accepting the OCR_RAW as HUMAN_CORRECTED without presenting it to the annotator. Use with caution — verify this is safe on a sample before enabling. Saves annotation time on easy lines.

---

## Notes for the Coding Agent

- All database writes that create GT or HUMAN_CORRECTED transcriptions/segmentations must set `immutable=1`. This is enforced by DB triggers as well, but do not rely on triggers alone — set the flag explicitly in application code.
- Never call `DELETE` or `UPDATE` on rows where `immutable=1`. Check before any write to these tables.
- All file paths stored in the DB are relative to `root_path`. Never store absolute paths. Use `config.resolve_path()` whenever constructing an actual filesystem path.
- The `processing_status` field on `pages` is the pipeline's state machine. Respect it. Add a helper function `can_process(page, stage)` that checks status and `force_reprocess` flag before any stage runs.
- Prefer SQLite transactions for multi-row writes (e.g. creating a segmentation + all its lines). Partial writes are worse than failed writes.
- All Flask routes that write to the DB should return JSON responses (not page redirects) so the frontend can handle success/error without a full page reload.
