# Inkwell V1 Execution Plan — Updated (2026-03-11)

**Status:** Active development plan incorporating multi-model evaluation strategy

---

## Implementation Snapshot (2026-03-11)

This section reflects what is currently implemented in code (not just planned):

- `/annotate` exists and is actively used for human correction.
- Annotation queue now samples **random unannotated lines** (not sequential per page).
- Main annotation view includes 3 stacked image views:
  - original crop,
  - 2x magnified crop,
  - context crop (`/annotate/api/context/<line_id>`) derived from page + polygon coordinates.
- Marker shorthand is standardized for speed:
  - `[ur]` = unreadable,
  - `[nt]` = not text,
  - `[?]` = uncertain.
- Marker buttons and Hungarian diacritic helper buttons are available in UI.
- Optional flags are supported and stored in `transcriptions.flag`:
  - `SEGMENTATION_ISSUE`,
  - `UNUSABLE_SEGMENTATION`,
  - `NOT_TEXT`.
- Review/edit flow exists:
  - `/annotate/review` list,
  - `/annotate/edit/<line_id>` editor,
  - `/annotate/api/update` endpoint.
- Progress stats were fixed to use **distinct line counts** (multi-model OCR rows no longer inflate counts).
- Progress is displayed as both `annotated/total` and percentage.
- Human annotation inserts now set `immutable=1`.

Current known state:

- preprocessing / page splitting / rotation were run on the full source set,
- line segmentation was run across the corpus,
- OCR infrastructure works and full-source OCR has now been run by the user,
- annotation is ongoing (roughly first 50 corrected lines collected),
- `dataset_splits` table exists but is not yet populated/used in annotation routing,
- annotation currently runs over all OCR-available lines in DB, independent of split assignment.

## New-session handoff summary

If a new LLM session starts, it should assume:

1. The annotation UI is functional and currently the main active workflow.
2. Human labels are being collected in `transcriptions` as immutable rows.
3. The next engineering tasks are **not** more annotation UX, but data-pipeline tasks around:
  - frozen page-level dataset splits,
  - GT export,
  - baseline evaluation,
  - model research / comparison,
  - fine-tuning.
4. Archived docs should be ignored unless explicitly needed for historical context.

---

## What's Complete

✅ Database schema with immutable triggers  
✅ Asset ingestion (idempotent, 498 images confirmed)  
✅ `/ingest` UI (orientation/layout confirmation with full UX)  
✅ Preprocessing (rotate, deskew, split doubles - quality validated)  
✅ Line segmentation (CV projection baseline - 798 pages, 15,965 lines)  
✅ OCR infrastructure (EasyOCR + TrOCR support, pluggable backend, resumable)  
✅ Visual sampling tool (`scripts/sample_ocr.py`)  
✅ Annotation UI, review/edit flow, helper buttons, flags, random queue  

**Current state:** pipeline works end-to-end; the active bottleneck is label collection and model selection quality.

---

## Remaining Work for V1

### **Phase 1 Wrap-Up**

#### 1.1 Export Pipeline ⏳
**Goal:** Generate text files with full lineage for downstream use.

**Implementation:**
- `inkwell/pipeline/export.py`
- CLI: `python scripts/run_pipeline.py export [--output-dir DIR]`
- Output format: one file per page or notebook with line references
- Include metadata: source image, page ID, line ID, transcription authority

**Done when:** Can generate readable text from any notebook with traceable lineage.

---

### **Phase 2: Multi-Model OCR + Human Correction Loop**

#### 2.1 Multiple OCR Backends 🎯
**Goal:** Compare multiple OCR models systematically, not just one.

**Approach:**
- Keep pluggable backend structure (already done)
- Add backends sequentially: TrOCR, PaddleOCR, Kraken
- Each backend stores transcriptions with unique `model_version` identifier
- Run same lines through multiple models separately (not in parallel)

**Implementation sketch:**
```bash
# Sequential runs with different backends
python scripts/run_pipeline.py ocr --model easyocr --force  # Already done
python scripts/run_pipeline.py ocr --model trocr --force
python scripts/run_pipeline.py ocr --model paddleocr --force
```

**Database strategy:**
- `transcriptions` table already supports multiple rows per line_id
- Filter by `transcription_type='OCR_AUTO'` + `model_version`
- Authority hierarchy: `GT > HUMAN_CORRECTED > OCR_AUTO` (best model chosen per line)

**Done when:** Can run 2-3 different OCR models on same lines and store results independently.

#### 2.2 Model Comparison Tool 📊
**Goal:** Quantify which OCR backend performs best.

**Metrics to track:**
- Character Error Rate (CER) vs human GT
- Word Error Rate (WER) vs human GT  
- Mean confidence by model
- Processing speed
- Per-model confusion patterns

**Implementation:**
- `scripts/compare_models.py` - generate comparison report
- Requires human GT lines collected first (see 2.4)

#### 2.3 Annotation UI (`/annotate`) 🖊️
**Goal:** Human correction loop with model comparison view.

**Core features:**
- Display line crop image
- Show **all OCR results** side-by-side (EasyOCR / TrOCR / etc.)
- Text input for correction
- Keyboard shortcuts: Enter=submit, Ctrl+1/2/3=copy model N result, Escape=skip
- Queue: lowest min-confidence across all models first
- Store as `HUMAN_CORRECTED` transcription (immutable=1)

**Nice-to-have:**
- Model agreement highlighting (green if 2+ models agree)
- Confidence badges per model
- Progress counter (target: 200+ corrected lines)

**Done when:** Can annotate 200 lines with human corrections stored immutably.

#### 2.4 Ground Truth Storage Strategy 🗂️
**Problem:** Segmentation changes break line_id references.

**Solution (position-based matching):**

Create new table:
```sql
CREATE TABLE ground_truth_lines (
  id INTEGER PRIMARY KEY,
  page_id INTEGER NOT NULL,
  vertical_center_y INTEGER NOT NULL,  -- Y-coordinate of line center
  horizontal_left INTEGER,              -- Left bound (optional)
  horizontal_right INTEGER,             -- Right bound (optional)
  text TEXT NOT NULL,
  source TEXT NOT NULL,                 -- 'HUMAN_CORRECTED' or 'GT'
  created_at TEXT NOT NULL,
  created_by TEXT NOT NULL,
  notes TEXT
);
```

**Matching logic:**
When segmentation changes, match GT to new lines by:
1. Same page_id
2. Vertical center within ±20px tolerance
3. Horizontal overlap >50%

**Migration from transcriptions:**
- Copy `HUMAN_CORRECTED` transcriptions to `ground_truth_lines`
- Calculate vertical_center_y from line polygon_coords
- Keep both tables (transcriptions for OCR history, GT for training)

**Done when:** GT survives segmentation reruns and auto-reassociates with new line boundaries.

#### 2.5 Simple Segmentation Fixes 🔧
**Approach:** Manual database edits + optional helper script, not a full UI.

**For rare bad segmentation:**
- Mark line as `skip=1` with `skip_reason='bad_segmentation'`
- Or: manually INSERT/UPDATE lines table with corrected polygon_coords
- Or: simple CLI tool to split/merge line records

**Done when:** Can handle <5% segmentation errors without blocking annotation work.

---

### **Phase 3: Multi-Model Fine-Tuning + Evaluation**

#### 3.1 Dataset Splits (Page-Level) 📑
**Goal:** Prevent train/val leakage from neighboring lines.

**Strategy:**
- Assign split **by page** (or notebook chunk), never by individual lines
- 70% train / 15% val / 15% test
- Store in `dataset_splits` table (already exists)
- Never reshuffle once assigned

**Implementation:**
```bash
python scripts/assign_splits.py --strategy page_random --seed 42
```

#### 3.2 Fine-Tuning Multiple Models 🏋️
**Goal:** Improve each OCR model with project-specific GT data.

**Approach:**
- Separate fine-tuning runs per model (not parallel)
- Each model uses same train/val split
- Store checkpoints with clear naming: `trocr_finetuned_200gt`, `easyocr_finetuned_200gt`

**Implementation sketch:**
```bash
# Fine-tune each model separately
python scripts/run_pipeline.py finetune --model trocr --checkpoint run1
python scripts/run_pipeline.py finetune --model paddleocr --checkpoint run1
```

**Data source:**
- Training data: `ground_truth_lines` where page_id in train split
- Validation: GT lines in val split
- Test: GT lines in test split (frozen, never used for training)

**Done when:** Can fine-tune 2-3 models and generate separate checkpoints.

#### 3.3 Model Evaluation Report 📈
**Goal:** Compare baseline vs fine-tuned models quantitatively.

**Report contents:**
- CER/WER per model on frozen test set
- Baseline vs post-fine-tune comparison
- Per-model confidence calibration
- Processing time (lines/second)
- Identify best model per metric

**Output format:** Markdown or HTML report with tables/charts

**Done when:** Clear data shows which model(s) to use for production OCR.

---

## Implementation Priorities (Immediate Next Steps)

### **Priority 1: Data discipline before training**
1. Assign page-level dataset splits and freeze them
2. Make annotation / export split-aware
3. Export GT-ready corrected data from immutable human labels

### **Priority 2: Model research and bake-off**
4. Research Hungarian handwriting-capable pretrained checkpoints
5. Run small bake-off on corrected lines (CER/WER)
6. Pick best base model for fine-tuning

### **Priority 3: Fine-tuning + evaluation**
7. Fine-tune selected base model on train split
8. Evaluate on frozen validation split
9. Compare against current baseline OCR

---

## Design Decisions

### Multi-Model Strategy
- **Sequential, not parallel:** Run one model at a time to keep code simple
- **Shared line crops:** All models process same segmented lines
- **Independent storage:** Each model's output stored separately with model_version tag
- **Unified evaluation:** Compare all models against same GT set

### Ground Truth Philosophy
- **Position-based, not ID-based:** GT survives segmentation changes via coordinate matching
- **Immutable once created:** GT records never updated, only supplemented
- **Separate from OCR history:** `ground_truth_lines` table distinct from `transcriptions`

### Annotation UX Priority
- **Speed over beauty:** Keyboard-focused, minimal clicks
- **Model comparison is core:** Show all model outputs to inform correction
- **Low-confidence first:** Queue by uncertainty (min confidence across models)

### Segmentation Correction Scope
- **Database edits acceptable:** Not building full segmentation UI for V1
- **<5% error rate OK:** Spend annotation time on text, not polygons
- **Helper scripts over UI:** CLI tools for rare manual fixes

---

## Out of Scope for V1

❌ Real-time multi-model parallel processing  
❌ Rich polygon editing UI  
❌ Automatic model ensemble/voting  
❌ Advanced active learning  
❌ Cloud/multi-user features  
❌ LLM-based rewriting or semantic correction  

---

## Success Criteria (V1 Complete)

✅ 200+ human-corrected GT lines stored in position-based format  
✅ 2-3 OCR models compared on same test set  
✅ At least one fine-tuned model shows measurable CER improvement  
✅ Export pipeline produces text with full lineage  
✅ GT survives segmentation parameter changes  
✅ Clear quantitative evidence of which model(s) work best  

---

## Next Session Starting Point

**Immediate action:** keep annotating toward ~200 corrected lines while implementing split assignment + GT export in parallel.

**Rationale:** annotation is already working; the next risk is not UX, but keeping data clean and evaluation leakage-free before training starts.
