# Inkwell — Development Master Plan (Single Source of Truth)

Date: 2026-03-09
Status: Canonical implementation plan for development
Supersedes: brainstorming-level execution details spread across earlier docs

---

## 1. Mission and scope

Inkwell is a local-first OCR pipeline for one fixed diary dataset. The goal is to produce faithful transcriptions with traceable lineage and iterative quality improvement.

### In scope (V1)
- ingest and normalize source assets,
- page-level processing status tracking,
- line segmentation and OCR,
- human correction loop,
- fine-tuning loop,
- text export with lineage.

### Out of scope (V1)
- rich segmentation editor,
- full region polygon tooling,
- advanced active-learning diversity,
- publication-focused LLM rewriting,
- multi-user/cloud features.

---

## 2. Non-negotiables

1. **Lineage is mandatory:** every exported line must map back to page and source image.
2. **Human-approved data is immutable:** no update/delete for immutable segmentation/transcription rows.
3. **Faithful source text:** preserve misspellings and dyslexic forms exactly.
4. **Reprocessing is explicit:** no silent overwrite of authoritative rows.
5. **Portable paths:** store relative paths in DB; root path configured centrally.

---

## 3. Reality-check conclusions and fixes

### Identified risks
- integration overload (too many complex components at once),
- optimistic evaluation from line-level random split leakage,
- premature UI scope,
- confidence misuse as auto-approval,
- schema over-expansion before value is proven.

### Adopted simplifications
- sequence work strictly: pipeline first, then annotation, then training loops, then optional UX,
- split train/val/test by page (or notebook chunk), never by individual neighboring lines,
- use uncertainty-only candidate ordering initially,
- keep V1 schema minimal and grow only when a feature is being implemented,
- no automatic promotion from OCR to human-approved in V1.

---

## 4. V1 system architecture

## 4.1 Components
- **SQLite DB**: single source of truth.
- **Pipeline scripts**: ingest, preprocess, segment, ocr, finetune, export.
- **Flask app**: `/ingest`, `/annotate`, plus annotation support routes (`/annotate/review`, `/annotate/edit/<line_id>`, `/annotate/api/*`).
- **Working directory**: derived images and line crops.

## 4.2 Data flow
1. Ingest source files into DB.
2. Confirm orientation/layout in minimal ingestion UI.
3. Preprocess pages.
4. Segment lines and generate crops.
5. OCR lines and store `OCR_RAW` transcriptions.
6. Annotate low-confidence lines as `HUMAN_CORRECTED` (immutable).
7. Fine-tune with train split; evaluate on frozen validation split.
8. Export highest-authority text.

---

## 5. Canonical V1 data model

Only these tables are required for V1 implementation:

- `config(key, value)`
- `notebooks(id, label, folder_name, date_start, date_end, notes)`
- `assets(id, notebook_id, filename, file_order, asset_type, should_ocr, notes)`
- `source_images(id, asset_id, orientation_detected, orientation_confirmed, layout_type, derived_image_path, notes)`
- `pages(id, source_image_id, side, page_type, processing_status, force_reprocess, notes)`
- `segmentations(id, page_id, segmentation_type, line_polygons, model_version, created_at, immutable)`
- `lines(id, page_id, segmentation_id, line_order, polygon_coords, crop_image_path, segmentation_confidence, skip, skip_reason)`
- `transcriptions(id, line_id, transcription_type, text, confidence, model_version, created_at, created_by, immutable)`
- `dataset_splits(id, page_id, split, assigned_at)`

### Required triggers
- block `UPDATE` on `transcriptions` where `immutable=1`
- block `DELETE` on `transcriptions` where `immutable=1`
- block `UPDATE` on `segmentations` where `immutable=1`
- block `DELETE` on `segmentations` where `immutable=1`

### Authority rules
- transcription authority: `GT > HUMAN_CORRECTED > OCR_RAW`
- segmentation authority: `GT > HUMAN_CORRECTED > AUTO`

---

## 6. Directory and module blueprint

Target implementation structure:

```text
inkwell/
  inkwell/
    db.py
    config.py
    pipeline/
      ingest.py
      preprocess.py
      segment.py
      ocr.py
      finetune.py
      export.py
    web/
      app.py
      routes/
        ingest.py
        annotate.py
  scripts/
    init_db.py
    run_pipeline.py
    set_root.py
  working/
    inkwell.db
    derived_images/
    line_crops/
```

---

## 7. Pipeline contract (stage-by-stage)

### Stage A — Init + ingest
**Goal:** DB initialized and assets registered in deterministic order.

**Inputs:** `notebooks_config.json` + source folders.

**Outputs:** rows in `notebooks`, `assets`, `source_images`, `pages`.

**Done when:** rerunning init/ingest is idempotent.

### Stage B — Preprocess
**Goal:** produce normalized page images for segmentation.

**Default ops:** rotate + light deskew; avoid aggressive thresholding as default.

**Outputs:** `derived_image_path` populated.

**Done when:** sample pages look stable and readable; no obvious clipping/destruction.

### Stage C — Segmentation
**Goal:** generate line boundaries and line crops.

**Engine:** Kraken baseline segmentation.

**Outputs:** `segmentations(AUTO)` + `lines` + crop image files.

**Done when:** line ordering and crop quality are acceptable on representative pages.

### Stage D — OCR
**Goal:** produce first-pass transcriptions for all non-skipped lines.

**Engine:** TrOCR base model.

**Outputs:** `transcriptions(OCR_RAW)` with confidence.

**Done when:** full notebook has exportable OCR text.

### Stage E — Annotation loop
**Goal:** convert low-confidence OCR lines into immutable human-corrected labels.

**Queue:** lowest confidence first (uncertainty-only).

**Outputs:** `transcriptions(HUMAN_CORRECTED, immutable=1)`.

**Done when:** at least 200 corrected lines collected across representative pages.

### Stage F — Fine-tuning + eval
**Goal:** improve model quality with project-specific labels.

**Split rule:** assign split by page once; never reshuffle.

**Outputs:** checkpoint + CER/WER on frozen validation set.

**Done when:** measurable improvement over baseline on validation set.

---

## 8. V1 web UX requirements

Only two routes are mandatory in V1.

### `/ingest`
- thumbnail list/grid,
- confirm orientation/layout,
- mark non-processable assets (`should_ocr=0`),
- quick bulk confirm action.

### `/annotate`
- display line crop + OCR guess,
- edit and submit corrected text,
- skip action,
- keyboard-first actions (submit/skip/next),
- progress indicator (corrected count).

No additional pages are required for V1 completion.

### Implemented deviations/decisions (2026-03-10)

- `transcription_type='OCR_AUTO'` is used in implementation (instead of `OCR_RAW` naming in older plan text).
- Annotation queue currently uses **random unannotated-line sampling**.
- Annotation helper routes are implemented and in use (`review`, `edit`, `update`, context image endpoint).
- Progress uses distinct line counts to avoid multi-model overcounting.
- Marker shorthand for annotation speed is standardized in UI:
  - `[ur]` unreadable,
  - `[nt]` not text,
  - `[?]` uncertain.

---

## 9. CLI contract

All pipeline entry points must support these arguments where relevant:

- `--root <path>` (override root path)
- `--page <id>` (single-page processing)
- `--force` (reprocess stage)
- `--model <checkpoint>` (OCR/fine-tune source model)

Required script calls:

```bash
python scripts/init_db.py --config notebooks_config.json
python scripts/run_pipeline.py ingest
python scripts/run_pipeline.py preprocess
python scripts/run_pipeline.py segment
python scripts/run_pipeline.py ocr --model microsoft/trocr-base-handwritten
python scripts/run_pipeline.py finetune --model <optional_checkpoint>
python scripts/run_pipeline.py export
```

---

## 10. Processing status model

Use this minimal linear state machine in V1:

`pending -> preprocessed -> segmented -> ocr_done -> reviewed`

Rules:
- each stage only advances one step,
- `--force` allows rerunning a stage,
- reruns append new AUTO/OCR rows; never mutate immutable authoritative rows.

---

## 11. Quality and evaluation policy

### Metrics to track
- CER (primary),
- WER (secondary),
- annotation throughput (lines/hour),
- OCR confidence distribution shift after fine-tune.

### Baseline quality checkpoints
1. baseline OCR metrics on validation split,
2. after first 200 corrections + fine-tune,
3. after each additional 100 corrections.

### Auto-accept policy
Disabled in V1. Consider only after calibration shows high precision at threshold.

---

## 12. Implementation phases and goals

## Phase 0 — Foundation (must complete first)
**Goal:** runnable project skeleton and DB integrity rules.

Deliverables:
- package structure,
- DB schema + immutable triggers,
- root-path config helpers,
- `init_db.py` and idempotent ingestion.

Exit criteria:
- schema created successfully,
- assets discovered and ordered,
- rerun does not duplicate rows.

## Phase 1 — First end-to-end OCR
**Goal:** produce OCR output from real pages.

Deliverables:
- preprocess,
- segment + crop generation,
- OCR inference + storage,
- plain text export with lineage references.

Exit criteria:
- at least one notebook fully processed to exported text.

## Phase 2 — Human correction loop
**Goal:** improve data quality and collect training labels.

Deliverables:
- `/annotate` route,
- low-confidence queue,
- immutable `HUMAN_CORRECTED` inserts.

Exit criteria:
- 200 corrected lines collected.

## Phase 3 — Fine-tune and validate
**Goal:** prove model improvement.

Deliverables:
- page-level split assignment,
- fine-tuning script,
- validation report (CER/WER).

Exit criteria:
- improved validation CER versus baseline.

## Phase 4 — Evidence-based upgrades (optional)
**Goal:** add complexity only where bottlenecks are measured.

Candidate upgrades:
- segmentation correction UI,
- region editing,
- diversity-aware active learning,
- viewer UX.

Exit criteria:
- each upgrade justified by explicit metric pain.

---

## 13. Decision gates for adding complexity

- Add segmentation editor only if >15% of pages need manual segmentation fixes.
- Add diversity scoring only if uncertainty queue over-concentrates on narrow subsets.
- Add region polygons only if page-level skip fails frequently on mixed pages.
- Enable auto-accept only if calibrated threshold precision is high on frozen validation.

If a gate condition is not met, do not implement the feature yet.

---

## 14. Engineering standards for this project

- keep changes focused and minimal,
- avoid adding tables/routes without an immediate consumer,
- use transactions for multi-row writes,
- treat pipeline scripts as idempotent,
- log stage summaries with counts and failure reasons,
- never silently discard data.

---

## 15. Definition of done (V1)

V1 is complete when all are true:

1. assets ingest and preprocessing are stable,
2. segmentation and OCR run end-to-end,
3. annotation UI stores immutable human-corrected rows,
4. fine-tuning runs from corrected data with page-level split,
5. validation CER improves after fine-tune,
6. export produces highest-authority text with lineage,
7. no immutable row can be updated/deleted.

---

## 16. Immediate execution checklist (start here)

1. implement Phase 0 schema/config/init tooling,
2. run ingest on full dataset,
3. build minimal `/ingest` for orientation/layout confirmation,
4. implement preprocess + segment + OCR,
5. export and inspect first notebook text,
6. continue annotation until first meaningful milestone (~200 corrections),
7. assign frozen page-level splits and export GT-ready data,
8. run first fine-tune and compare CER,
9. decide next step using decision gates.

This checklist is the default operating plan unless explicitly revised.
