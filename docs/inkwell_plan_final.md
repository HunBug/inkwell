# Inkwell — Current Plan (Canonical)

Date: 2026-03-14  
Status: Active source of truth for architecture + roadmap.

## 1) Goal

Build a repeatable OCR improvement loop for handwritten Hungarian diaries:

1. segment + OCR full corpus,
2. annotate prioritized lines,
3. fine-tune,
4. evaluate on frozen split,
5. repeat.

Primary success metric: lower CER/WER on fixed validation/test sets.

## 2) Current system (implemented)

- SQLite DB as source of truth (`working/inkwell.db`)
- Web UI:
  - `/annotate` (+ queue mode from suggestions files)
  - `/annotate/review`, `/annotate/edit/<line_id>`
  - `/jobs` (automation, code sync, eval submit, pool inference launcher)
- GPU worker queue over shared folder:
  - `finetune`, `eval`, `infer_pool`
- Dataset/export tooling:
  - GT export for train/val/test
  - unlabeled full-pool export (`unlabeled_pool/`)
- Full-pool inference:
  - fine-tuned checkpoint runs over ~15k unlabeled lines
  - results stored in job artifact file (`pool_predictions.jsonl`)
- Annotation suggestion generation:
  - `scripts/pick_next_samples.py`
  - consumes eval + pool predictions
  - outputs `working/suggestions/next_samples_*.jsonl`

## 3) Non-negotiables

1. Keep annotation mapping stable (`line_id` remains authoritative for existing labels).
2. Prefer slight context noise over clipping current-line glyphs (especially accents).
3. Keep large inference outputs file-based (do not bulk-write 10k+ predictions into DB each run).
4. Keep evaluation split frozen for meaningful comparisons.
5. Preserve immutable human labels; no silent overwrite.

## 4) Operating loop

Use this loop continuously:

1. Annotate from latest suggestions queue (`/annotate` queue mode).
2. Export + train (`run_automation.py` or `/jobs` automation button).
3. Run baseline/fine-tuned eval (`/jobs` eval controls).
4. Run full unlabeled inference (`/jobs` pool button).
5. Generate next suggestions (`pick_next_samples.py`).
6. Repeat.

## 5) Parallel work policy

While annotation is ongoing, parallel engineering should focus on:

- crop quality improvements without breaking ID matching,
- better candidate ranking quality,
- faster/safer operations (sync/export/infer),
- compact, up-to-date documentation.

Avoid broad UI expansion until loop quality is stable.

## 6) Active technical decisions

### 6.1 Crop quality

- Bottom margin is currently acceptable.
- Top margin should be slightly larger to reduce accent clipping.
- Safe approach:
  - recrop existing lines in place (prefer unannotated-only by default),
  - keep segmentation IDs/line IDs unchanged.

### 6.2 Full-pool inference storage

Store in job artifacts only:

- `jobs/infer_pool_.../pool_predictions.jsonl`
- `jobs/infer_pool_.../result.json`

Do not insert all predictions into DB.

### 6.3 Suggestions quality

Current ranking blends:

- OCR confidence,
- eval-derived hard-char heuristics,
- OCR-vs-finetuned disagreement from full-pool predictions,
- diversity caps per page/notebook.

This should keep evolving with observed annotation value.

## 7) Near-term roadmap

### Next (high priority)

1. Improve suggestion quality filters (reduce symbol/noise-heavy picks).
2. Add lightweight quality-bucket sampling (hard + medium + clean mix).
3. Validate top-margin recrop impact with a short benchmark run.

### Then (medium priority)

4. Optional better segmenter experiment on GPU (versioned, non-destructive).
5. Compare segmentation variants on fixed sample pages before broad adoption.

### Later (low priority)

6. Dedicated error-analysis page once loop logic is stable.

## 8) Definition of “good state”

- Annotation queue consistently yields useful lines.
- Fine-tune runs are stable and fast.
- CER/WER trend improves round-over-round.
- Docs are concise and accurate for new-session handoff.
