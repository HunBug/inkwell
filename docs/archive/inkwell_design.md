# Inkwell — Design Document

> **Project name:** Inkwell  
> **Repository:** `inkwell`  
> **Scope:** Personal hobby project. Fixed input dataset (one person's handwritten diaries). Not a general product.  
> **Goal:** OCR a set of scanned handwritten Hungarian diary notebooks with maximum accuracy, store all data with full lineage, and provide a simple web viewer.

---

## 1. Project Overview

Inkwell is a personal pipeline for digitising handwritten diary notebooks. It combines classical computer vision, neural OCR (TrOCR), active learning, and a lightweight web UI. The entire system is self-contained: one SQLite database, one local web app, one Python venv.

The pipeline is intentionally built as composable "lego blocks" — each stage is a separate script or module that reads from and writes to the central database. Stages can be re-run independently. No stage silently overwrites authoritative human-verified data.

### Source material

- ~500 JPG scans at consistent DPI, mix of colour and greyscale
- Some scans contain a single diary page, some contain a double-page spread
- Some scans are 90° rotated (not yet corrected)
- Folders also contain scans of attachments/photos that should not be OCR'd
- 8 folders total, one per notebook
- Filename format: `IMG_YYYYMMDD_NNNN.jpg` — the counter `NNNN` gives page order within a notebook, the date is scan date (irrelevant to content)
- Content date range: approximately ages 10–18 (roughly 1990s–2000s)
- Language: Hungarian (~99%), occasional German/English/French words
- Handwriting: cursive, variable quality, changes significantly across years and mood. Writer has dyslexic tendencies (letter transpositions, substitutions, omissions). This is considered correct source text, not errors to fix.

---

## 2. Design Principles

**Lineage is non-negotiable.** Every piece of output text must be traceable back to a specific line crop, which traces back to a specific page, which traces back to a specific source file. This is enforced at the data model level from day one.

**Authoritative data is immutable.** Once a transcription or segmentation is marked as Ground Truth (GT) or Human Corrected, the pipeline never overwrites it. New model runs append new rows; they never touch existing GT rows.

**Single source of truth.** One SQLite database contains all state. Working image files live in a designated working directory, with paths stored in the DB. Original scans are never modified.

**Root path portability.** The database stores relative paths. The root folder path is stored once as a config value and can be overridden via CLI argument. Moving the entire dataset requires changing one value.

**Reprocessing is safe.** Every page has a `processing_status` field. Pipelines skip pages that are already processed unless explicitly forced. This enables partial reprocessing of single pages after corrections.

**Dyslexic source text is preserved faithfully.** The OCR goal is faithful transcription of what is written, including misspellings, transpositions, and wrong letters. A separate post-processing step (LLM, out of scope for now) produces a "publication quality" version. The database stores both, independently.

---

## 3. Technology Choices and Rationale

### OCR engine: TrOCR (microsoft/trocr-base-handwritten)

TrOCR is a transformer-based OCR model available on HuggingFace. It is pretrained on large handwriting datasets and can be fine-tuned on custom data using the standard HuggingFace `Trainer` API.

`trocr-base` is used because `trocr-large` requires ~10GB VRAM. The available GPU (RTX 2050/2060, 4–6GB VRAM) fits `trocr-base` with batch size 4–8. For a final high-quality training run, Google Colab Pro (~$10/month) can run `trocr-large` on a T4 GPU.

TrOCR works on individual line images, not full pages. This is why line segmentation is a prerequisite step.

Hungarian diacritics (á, é, ő, ű, etc.) are underrepresented in TrOCR's pretraining data. Fine-tuning on this specific handwriting directly addresses this.

### Segmentation: Kraken

Kraken is an OCR framework designed for historical and manuscript documents. Its segmentation model (BaselineDet) is a trained neural network that outputs baseline polygons per text line. It handles curved lines, variable line spacing, and non-standard layouts better than classical projection-based methods.

Classical CV line segmentation (horizontal projection profiles) was considered and rejected because it breaks on slightly curved lines, drifting baselines, variable spacing, and inline non-text elements — all of which occur in this dataset.

Kraken's segmenter is the primary tool. If it fails on unusual pages, YOLOv8 or Detectron2 are available escalation paths but likely unnecessary.

**Known issue:** Kraken's BaselineDet, trained primarily on Latin/English manuscripts, sometimes attaches Hungarian double acute accents (ő, ű, í) to the wrong line as punctuation. Mitigation: add generous vertical padding (configurable, default 15%) to all line crops before feeding to TrOCR. If this is insufficient, Kraken's segmenter can itself be fine-tuned on corrected segmentation ground truth (separate training pass, ~20–30 corrected pages).

### Preprocessing: Classical CV (OpenCV + PIL)

For binarisation, deskew, and contrast normalisation, classical CV is fast, deterministic, and requires no training data.

`unpaper` (CLI tool) is available for bleed-through cleanup if ink from the reverse side of a page bleeds through. This is an escalation option if Otsu/adaptive threshold binarisation is insufficient.

### Active learning

After an initial seed fine-tune, the model scores all unreviewed lines by confidence. Low-confidence lines are surfaced first for human annotation. This maximises the information gained per hour of annotation labour.

The active learner also applies a diversity penalty: lines visually similar to already-labelled lines are deprioritised. This prevents the system from obsessively surfacing lines from one bad page while ignoring other handwriting periods.

Implementation: ~200 lines of Python using existing model confidence outputs. No external active learning framework needed.

### Database: SQLite

The dataset is small (≤500 pages, a few thousand lines). SQLite is sufficient, portable, and requires no server. The entire project state is one file.

### Web UI: Flask + vanilla JS

Simple, local-only, no build step. Serves the annotation UI, ingestion review UI, segmentation review UI, and viewer. All sections are routes in one Flask app on one port.

### Python environment: venv + requirements.txt

Standard, reproducible, Arch Linux compatible.

---

## 4. Data Model

All tables are in one SQLite database. The database is the single source of truth. Working image files (derived/cropped images) are stored on disk with paths in the DB.

```sql
-- Configuration: one row per key
config (
  key   TEXT PRIMARY KEY,
  value TEXT
)
-- Key: 'root_path' — absolute path to the folder containing all notebook folders.
-- Override at runtime with CLI argument --root /new/path.

-- One row per notebook folder
notebooks (
  id           INTEGER PRIMARY KEY,
  label        TEXT,           -- human-readable name, e.g. "Blue notebook 1993"
  folder_name  TEXT,           -- folder name relative to root_path
  date_start   TEXT,           -- approximate, e.g. "1993-09"
  date_end     TEXT,           -- approximate, e.g. "1994-06"
  notes        TEXT
)

-- One row per file in the notebook folders
assets (
  id               INTEGER PRIMARY KEY,
  notebook_id      INTEGER REFERENCES notebooks(id),
  filename         TEXT,       -- filename only, no path
  file_order       INTEGER,    -- extracted from filename counter for sorting
  asset_type       TEXT,       -- DIARY_PAGE | ATTACHMENT | EXTERNAL_DOC
  should_ocr       BOOLEAN DEFAULT 1,
  notes            TEXT
)
-- asset_type rationale:
--   DIARY_PAGE: standard diary content, runs through full pipeline
--   ATTACHMENT: scanned photo/document not from diary, shown in viewer but not OCR'd
--   EXTERNAL_DOC: text provided externally (see external_content table)
-- should_ocr=0 assets are stored in the DB and shown in the viewer at the correct
-- sequence position, but skipped by all processing stages.

-- Derived/processed version of an asset
source_images (
  id                      INTEGER PRIMARY KEY,
  asset_id                INTEGER REFERENCES assets(id),
  orientation_detected    INTEGER,  -- degrees: 0, 90, 180, 270
  orientation_confirmed   INTEGER,  -- confirmed by human (may differ from detected)
  layout_type             TEXT,     -- SINGLE | DOUBLE
  split_overlap_px        INTEGER DEFAULT 50,
  derived_image_path      TEXT,     -- path relative to root_path, deskewed/rotated working copy
  is_grayscale            BOOLEAN,
  notes                   TEXT
)

-- One logical page (SINGLE layout = 1 page, DOUBLE = 2 pages)
pages (
  id                      INTEGER PRIMARY KEY,
  source_image_id         INTEGER REFERENCES source_images(id),
  spread_id               INTEGER,  -- links two pages that came from one double-page scan
  side                    TEXT,     -- LEFT | RIGHT | FULL
  page_type               TEXT DEFAULT 'text',  -- text | drawing | mixed | blank | skip
  handwriting_profile_id  INTEGER REFERENCES handwriting_profiles(id),
  approximate_date        TEXT,     -- optional, e.g. "1994-03"
  date_confidence         TEXT,     -- LOW | MEDIUM | HIGH
  processing_status       TEXT DEFAULT 'pending',
  -- pending | segmented | ocr_complete | reviewed
  force_reprocess         BOOLEAN DEFAULT 0,
  notes                   TEXT
)

-- Named, reusable regions on a page (skip areas, special format areas, etc.)
page_regions (
  id              INTEGER PRIMARY KEY,
  page_id         INTEGER REFERENCES pages(id),
  region_type     TEXT,
  -- SKIP | DATE_MARKER | MARGIN_NOTE | DRAWING | STICKER | DAMAGED | UNREADABLE
  behavior        TEXT,
  -- SKIP_OCR: lines within this region are not fed to OCR
  -- PROCESS_SEPARATELY: region is treated as its own OCR task (future)
  -- FLAG_ONLY: lines are transcribed normally but tagged as overlapping this region
  polygon_coords  TEXT,  -- JSON array of [x,y] points
  flagged_by      TEXT,  -- AUTO | ANNOTATOR
  notes           TEXT
)
-- Regions can overlap. A line is matched against all regions it intersects.
-- The most restrictive behavior wins (SKIP_OCR > PROCESS_SEPARATELY > FLAG_ONLY).

-- Segmentation result for a page (append-only for GT/HUMAN_CORRECTED)
segmentations (
  id                  INTEGER PRIMARY KEY,
  page_id             INTEGER REFERENCES pages(id),
  segmentation_type   TEXT,  -- AUTO | HUMAN_CORRECTED | GT
  line_polygons       TEXT,  -- JSON: [{line_order, polygon: [[x,y],...], line_type}, ...]
  model_version       TEXT,
  created_at          TEXT,
  immutable           BOOLEAN DEFAULT 0
)
-- When pipeline needs segmentation for a page, it uses the highest-authority row:
-- GT > HUMAN_CORRECTED > AUTO
-- Reprocessing a page adds a new AUTO row; GT/HUMAN_CORRECTED rows are never touched.

-- One text line extracted from a page
lines (
  id                        INTEGER PRIMARY KEY,
  page_id                   INTEGER REFERENCES pages(id),
  segmentation_id           INTEGER REFERENCES segmentations(id),
  line_order                INTEGER,
  line_type                 TEXT DEFAULT 'BODY',
  -- BODY | MARGIN | DATE_MARKER | INFERRED | CROSSES_SPLIT
  polygon_coords            TEXT,  -- JSON [[x,y],...]
  crop_image_path           TEXT,  -- path to the cropped line image
  segmentation_confidence   REAL,
  skip                      BOOLEAN DEFAULT 0,
  skip_reason               TEXT,
  crosses_split             BOOLEAN DEFAULT 0,
  region_flags              TEXT   -- JSON list of page_region ids that overlap this line
)
-- INFERRED lines: annotator typed what should be there even though it is not
-- visible on the scan (damaged page, bad scan). These are still linked to the page
-- but crop_image_path may be null or point to the relevant damaged region.

-- All transcription versions for a line (append-only for GT/HUMAN_CORRECTED)
transcriptions (
  id                    INTEGER PRIMARY KEY,
  line_id               INTEGER REFERENCES lines(id),
  transcription_type    TEXT,
  -- OCR_RAW: direct model output, unreviewed
  -- HUMAN_CORRECTED: human reviewed, not formally GT
  -- GT: human verified, canonical, immutable
  -- INFERRED: annotator provided text not visible on scan
  -- LLM_POSTPROCESSED: LLM cleaned version (future)
  -- EXTERNAL: provided from outside pipeline
  text                  TEXT,
  -- Inline annotation conventions:
  --   [??] = unrecognisable span (text present but unreadable)
  --   [NT] = not-text inline element (drawing, sticker, symbol mid-line)
  confidence            REAL,
  model_version         TEXT,
  word_flags            TEXT,  -- JSON: [{word, position, type, verified_intentional}]
  created_at            TEXT,
  created_by            TEXT,  -- 'pipeline' | 'annotator'
  immutable             BOOLEAN DEFAULT 0
)
-- word_flags rationale: Hungarian dictionary can flag suspicious words.
-- Annotator confirms intentional misspellings (dyslexia). verified_intentional=true
-- means: "yes, that is what is written, preserve it."
-- The LLM postprocessing step receives word_flags and knows not to blindly correct
-- verified intentional misspellings.

-- Handwriting style profiles
handwriting_profiles (
  id                INTEGER PRIMARY KEY,
  label             TEXT,       -- e.g. "Early childhood", "Fast casual"
  description       TEXT,
  date_range_approx TEXT,
  notes             TEXT
)
-- Default: ~5 profiles, assigned at notebook level, overridable at page level,
-- further overridable at line level (via lines.handwriting_profile_id if needed).
-- Profiles are informational, not absolute. They help the active learner ensure
-- coverage across different handwriting styles.

-- Text provided from outside the pipeline (for ATTACHMENT/EXTERNAL_DOC assets)
external_content (
  id            INTEGER PRIMARY KEY,
  asset_id      INTEGER REFERENCES assets(id),
  content_type  TEXT,    -- TEXT | METADATA
  content_text  TEXT,
  source_tool   TEXT,    -- e.g. "Adobe Acrobat", "manual"
  import_date   TEXT
)

-- Dataset split assignment (set once, never changed for val/test)
dataset_splits (
  id           INTEGER PRIMARY KEY,
  line_id      INTEGER REFERENCES lines(id),
  split        TEXT,    -- train | val | test | unlabeled
  assigned_at  TEXT
)
-- Standard split: 80/10/10 train/val/test, random assignment.
-- val and test splits are frozen after initial assignment.
-- The test split is only used for final evaluation, never for training decisions.

-- Active learning tracking
active_learning_sessions (
  id              INTEGER PRIMARY KEY,
  started_at      TEXT,
  model_version   TEXT,
  lines_scored    INTEGER,
  lines_annotated INTEGER,
  notes           TEXT
)

active_learning_candidates (
  id              INTEGER PRIMARY KEY,
  session_id      INTEGER REFERENCES active_learning_sessions(id),
  line_id         INTEGER REFERENCES lines(id),
  uncertainty     REAL,    -- model confidence score (lower = more uncertain)
  diversity_score REAL,    -- distance from already-labelled lines
  priority_score  REAL,    -- combined score used for sorting
  presented       BOOLEAN DEFAULT 0,
  skipped         BOOLEAN DEFAULT 0
)
-- Lines annotator chose to skip (unreadable even to human) are marked skipped=1.
-- Skipped lines are presented again after more context lines around them are labelled.
```

---

## 5. Inline Annotation Conventions

Within transcription text, two special markers are used:

| Marker | Meaning |
|--------|---------|
| `[??]` | Unrecognisable span — text is present but neither human nor model can read it |
| `[NT]` | Not-text inline element — a drawing, sticker, or symbol embedded mid-line |

These markers are intentionally minimal. They provide signal for training (the model learns these regions are unpredictable) and for the viewer (rendered with a visual indicator). No other inline markup is used.

---

## 6. Immutability Rules

These rules are enforced at the application layer. Triggers in SQLite enforce them at the DB layer as a second line of defence.

1. A transcription with `transcription_type IN ('GT', 'HUMAN_CORRECTED')` and `immutable=1` cannot be updated or deleted.
2. A segmentation with `segmentation_type IN ('GT', 'HUMAN_CORRECTED')` and `immutable=1` cannot be updated or deleted.
3. When a human confirms a transcription during annotation, it is written with the appropriate type and `immutable=1` immediately.
4. When the pipeline runs OCR on a page that already has a GT/HUMAN_CORRECTED transcription for a line, it adds a new `OCR_RAW` row. It does not touch the existing authoritative row.
5. The output layer always displays the highest-authority transcription: GT > HUMAN_CORRECTED > OCR_RAW.

---

## 7. Processing Pipeline Stages

Each stage is a separate script. Stages are idempotent — running twice produces the same result. Each stage checks `processing_status` and skips pages that are already past that stage unless `force_reprocess=1` is set on the page.

```
Stage 0: Ingest
  Read notebook config JSON → populate notebooks + assets tables
  Detect orientation (Tesseract OSD)
  Detect single/double layout (heuristic: width/height ratio)
  Create source_images rows
  Flag non-diary assets for human review

Stage 1: Ingestion Review (UI)
  Human confirms/corrects orientation
  Human confirms single/double layout
  Human flags ATTACHMENT assets (should_ocr=0)
  Human draws page_regions (SKIP, DATE_MARKER, etc.)
  page.processing_status → 'ingested'

Stage 2: Preprocessing
  For each confirmed source_image:
    Rotate to correct orientation
    Deskew
    Binarise (adaptive threshold)
    Save to working directory
  Updates source_images.derived_image_path

Stage 3: Segmentation
  Kraken BaselineDet on each preprocessed page image
  Store result as segmentations row (type=AUTO)
  Crop line images with vertical padding (configurable, default 15%)
  Save crop images to working directory
  Create lines rows
  Apply page_regions: lines within SKIP_OCR regions get skip=1
  page.processing_status → 'segmented'

Stage 4: Segmentation Review (UI) [optional per page]
  Human views auto segmentation overlaid on page image
  Human corrects line boundaries
  Stores new segmentations row (type=GT or HUMAN_CORRECTED, immutable=1)
  Regenerates line crops for corrected lines

Stage 5: Initial OCR
  TrOCR inference on all non-skipped line crops
  Stores transcriptions rows (type=OCR_RAW)
  page.processing_status → 'ocr_complete'

Stage 6: Active Learning Loop (UI — main annotation stage)
  Score all unlabelled lines by uncertainty + diversity
  Surface highest-priority lines to annotator
  Annotator sees: line crop image + pre-filled OCR guess
  Annotator corrects text, marks [??] and [NT] spans
  On submit: store as HUMAN_CORRECTED (immutable=1), update dataset_splits
  On skip: mark active_learning_candidates.skipped=1
  After N annotations: trigger fine-tune run
  Repeat until accuracy is satisfactory

Stage 7: Fine-tuning (CLI)
  Load all GT + HUMAN_CORRECTED transcriptions from train split
  Fine-tune TrOCR-base (or trocr-large on Colab)
  Save checkpoint with version string
  Re-run Stage 5 on unlabelled pages with new model version

Stage 8: Export / Viewer
  Query DB for highest-authority transcription per line
  Render as text, JSON, or serve via web viewer
```

---

## 8. Active Learning Strategy

**Uncertainty scoring:** TrOCR outputs a probability per output token. Average log-probability across the sequence gives a line-level confidence score. Low score = uncertain line.

**Diversity scoring:** Embed each line crop using a simple image feature (histogram or small CNN embedding). Compute distance from each candidate line to its nearest already-labelled neighbour. Lines far from any labelled line get a diversity bonus.

**Combined priority:** `priority = uncertainty_weight * (1 - confidence) + diversity_weight * diversity_score`. Default weights: 0.7 / 0.3. Tunable.

**Seed set:** ~200 lines selected randomly across all notebooks. Ensures initial coverage before uncertainty scoring is meaningful.

**Training data split assignment:** Happens at annotation time. As each line is labelled, it is randomly assigned to train/val/test at 80/10/10. Val and test splits are frozen after assignment and never change.

**Handwriting profile coverage:** The active learner is aware of handwriting profile assignments. If a profile has fewer than a threshold of labelled lines, it gets a priority bonus. This prevents the model from ignoring rare profiles.

**Skip handling:** Annotator can skip a line they cannot read. Skipped lines are re-presented after surrounding context lines have been labelled (sometimes context helps decode an ambiguous word). If still unreadable, annotator marks it `[??]` for the whole line.

---

## 9. Special Page and Region Handling

### Page types

| Type | Behaviour |
|------|-----------|
| `text` | Normal pipeline |
| `drawing` | Segmentation skipped, page shown as image in viewer |
| `mixed` | Segmentation runs, non-text regions have SKIP_OCR behavior |
| `blank` | No processing |
| `skip` | Excluded entirely |

Auto-detection: pages where Kraken segmentation produces fewer than 3 lines, or lines with unusually wide/irregular polygons, are flagged as candidate `drawing` or `mixed` pages. Human confirms in the ingestion review UI.

### Region types and behaviors

| Region type | Default behavior |
|-------------|-----------------|
| `SKIP` | SKIP_OCR |
| `DATE_MARKER` | FLAG_ONLY |
| `MARGIN_NOTE` | FLAG_ONLY |
| `DRAWING` | SKIP_OCR |
| `STICKER` | SKIP_OCR |
| `DAMAGED` | FLAG_ONLY |
| `UNREADABLE` | SKIP_OCR |

Default behaviors can be overridden per region instance.

Regions are drawn in the ingestion review UI using a polygon drawing tool on the page image. Regions are stored as JSON polygon coordinates in `page_regions.polygon_coords`. Lines are matched against regions at segmentation time by polygon intersection.

---

## 10. UI Components

One Flask application, multiple routes. Runs locally on one port. Keyboard shortcuts for all annotation actions.

### 10.1 Ingestion Review (`/ingest`)

Used once at project start, then occasionally for corrections.

- Grid of source image thumbnails
- Per-image: orientation selector (0/90/180/270), layout selector (single/double), asset type selector
- Flag non-diary files with one click
- Link to region editor for each page

### 10.2 Region Editor (`/regions/<page_id>`)

- Full page image display
- Draw polygon regions (click to add points, close polygon)
- Select region type and behavior from dropdown
- Existing regions shown as coloured overlays
- Delete region button

### 10.3 Segmentation Review (`/segmentation/<page_id>`)

- Page image with line polygons overlaid
- Click polygon to select and drag corners to correct
- Add new line (draw rectangle/quad)
- Delete line
- Approve segmentation (saves as GT, immutable)

Simple quad editor (4-point bounding quadrilateral per line) rather than full polygon editor. Sufficient for diary pages where lines are roughly linear.

### 10.4 Annotation / Active Learning Loop (`/annotate`)

The core daily-use UI.

- Large display of line crop image
- Text input pre-filled with model's OCR guess
- Keyboard shortcuts: Tab to submit and go to next, Shift+Tab to go back, Escape to skip
- Progress bar: labelled / total unlabelled
- Confidence score shown (so annotator knows when model is very uncertain vs. just slightly off)
- Hungarian spell-check hint: suspicious words highlighted, annotator dismisses or confirms as intentional

### 10.5 Viewer (`/view`)

- Navigate by notebook and page
- Text view alongside scan image (side by side)
- Click any line in text view to highlight the corresponding crop on the scan
- Toggle between OCR_RAW, HUMAN_CORRECTED, and (future) LLM_POSTPROCESSED versions
- Search across all transcribed text
- Pages with `page_type != text` show image only
- Lines with `[??]` rendered with a visual indicator
- Lines with `[NT]` rendered with a visual indicator

---

## 11. File and Folder Structure

```
inkwell/
├── README.md
├── requirements.txt
├── notebooks_config.json        # Initial notebook metadata (one-time input)
│
├── inkwell/                     # Python package
│   ├── __init__.py
│   ├── db.py                    # DB connection, schema creation, migrations
│   ├── config.py                # Root path resolution, config table access
│   │
│   ├── pipeline/
│   │   ├── ingest.py            # Stage 0: read config, populate DB, detect orientation
│   │   ├── preprocess.py        # Stage 2: deskew, binarise, save working images
│   │   ├── segment.py           # Stage 3: Kraken segmentation, line crops
│   │   ├── ocr.py               # Stage 5: TrOCR inference, store OCR_RAW
│   │   ├── active_learn.py      # Stage 6: score lines, manage candidates
│   │   └── finetune.py          # Stage 7: fine-tune TrOCR from GT/HUMAN_CORRECTED data
│   │
│   ├── web/
│   │   ├── app.py               # Flask app, route registration
│   │   ├── routes/
│   │   │   ├── ingest.py
│   │   │   ├── regions.py
│   │   │   ├── segmentation.py
│   │   │   ├── annotate.py
│   │   │   └── viewer.py
│   │   └── templates/           # Jinja2 HTML templates
│   │       ├── base.html
│   │       ├── ingest.html
│   │       ├── regions.html
│   │       ├── segmentation.html
│   │       ├── annotate.html
│   │       └── viewer.html
│   │
│   └── export.py                # Query DB, export to text/JSON
│
├── scripts/
│   ├── init_db.py               # Create schema, import notebooks_config.json
│   ├── run_pipeline.py          # CLI runner for pipeline stages
│   └── set_root.py              # Update root_path in config table
│
└── working/                     # Generated files, not committed to git
    ├── inkwell.db
    ├── derived_images/          # Deskewed/rotated working copies
    └── line_crops/              # Individual line crop images
```

---

## 12. Configuration

### `notebooks_config.json` (initial input, one-time)

```json
{
  "root_path": "/home/user/diaries",
  "notebooks": [
    {
      "folder_name": "notebook_01",
      "label": "First diary",
      "date_start": "1993-09",
      "date_end": "1994-06",
      "handwriting_profile": "Early childhood",
      "notes": "Blue notebook, small format"
    }
  ],
  "handwriting_profiles": [
    { "label": "Early childhood", "description": "Age ~10, very unformed" },
    { "label": "Mid school", "description": "Age ~12-13, developing" },
    { "label": "Teen careful", "description": "Slow, readable writing" },
    { "label": "Teen fast", "description": "Quick, compressed, less readable" },
    { "label": "Late teen", "description": "Age ~17-18, more consistent" }
  ]
}
```

After `init_db.py` runs, this file is no longer needed. All data lives in the DB. The `root_path` can be changed at any time with `python scripts/set_root.py /new/path`.

### Runtime CLI arguments

All pipeline scripts accept:
- `--root /path/to/diaries` — override root_path from DB
- `--page <page_id>` — process only this page (partial reprocessing)
- `--force` — reprocess even if status is already past this stage
- `--model <checkpoint_path>` — use specific model checkpoint

---

## 13. Out of Scope (for now)

- LLM postprocessing for publication-quality text
- Per-page date detection and calendar navigation in the viewer
- Segmenter fine-tuning (escalation option if padding fix is insufficient)
- Separate fine-tune checkpoints per handwriting era
- Any form of cloud sync or multi-user access

The database schema already accommodates all of these. Adding them later requires writing new code but no schema migrations.
