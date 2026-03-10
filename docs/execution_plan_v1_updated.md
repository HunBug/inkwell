# Inkwell V1 Execution Plan — Updated (2026-03-10)

**Status:** Active development plan incorporating multi-model evaluation strategy

---

## What's Complete (Phase 0 + Phase 1 partial)

✅ Database schema with immutable triggers  
✅ Asset ingestion (idempotent, 498 images confirmed)  
✅ `/ingest` UI (orientation/layout confirmation with full UX)  
✅ Preprocessing (rotate, deskew, split doubles - quality validated)  
✅ Line segmentation (CV projection baseline - 798 pages, 15,965 lines)  
✅ OCR infrastructure (EasyOCR baseline - pluggable backend, resumable)  
✅ Visual sampling tool (`scripts/sample_ocr.py`)  

**Current state:** EasyOCR baseline quality is poor but pipeline works end-to-end.

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

### **Priority 1: Export + Multi-Model OCR Setup**
1. Implement export pipeline (simple text output)
2. Add TrOCR backend to `inkwell/pipeline/ocr.py`
3. Run TrOCR on same 50 lines as EasyOCR baseline
4. Generate comparison sample HTML showing both model outputs
5. Assess if TrOCR is worth pursuing or try PaddleOCR

### **Priority 2: GT Storage + Annotation UI**
6. Implement `ground_truth_lines` table and matching logic
7. Build `/annotate` route (line crop + multi-model results + text input)
8. Annotate 50 lines as proof-of-concept
9. Verify GT matching survives segmentation parameter tweaks

### **Priority 3: Fine-Tuning + Evaluation**
10. Assign page-level dataset splits
11. Fine-tune best-performing model(s) from Priority 1
12. Generate evaluation report comparing baseline vs fine-tuned
13. Decide on production OCR configuration

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

**Immediate action:** Implement TrOCR backend and run side-by-side comparison with EasyOCR on 50 lines to assess if it's worth the investment before building annotation UI.

**Rationale:** No point building multi-model annotation UI if TrOCR isn't meaningfully better than current EasyOCR baseline. Validate model quality before committing to full annotation loop.
