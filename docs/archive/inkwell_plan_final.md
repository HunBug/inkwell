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
  - `/annotate` (+ queue mode from suggestions files; shows latest OCR predictions with DB fallback)
  - `/annotate/review`, `/annotate/edit/<line_id>`
  - `/jobs` (automation, code sync, eval submit, pool inference, suggestions generation)
  - `/jobs/results` (cleaner view of experiment + eval trails by dataset)
- GPU worker queue over shared folder:
  - `finetune`, `eval`, `infer_pool`
- Dataset/export tooling:
  - GT export for train/val/test with configurable text-marker policy
  - unlabeled full-pool export (`unlabeled_pool/`)
  - policy persistence in dataset manifest
- Full-pool inference:
  - fine-tuned checkpoint runs over ~15k unlabeled lines
  - results stored in job artifact file (`pool_predictions.jsonl`)
- Annotation suggestion generation:
  - one-click button on `/jobs` or manual `scripts/pick_next_samples.py`
  - consumes eval + pool predictions + hard-char analysis
  - outputs `working/suggestions/next_samples_*.jsonl` (instantly loadable in Annotate)

## 3) Non-negotiables

1. Keep annotation mapping stable (`line_id` remains authoritative for existing labels).
2. Prefer slight context noise over clipping current-line glyphs (especially accents).
3. Keep large inference outputs file-based (do not bulk-write 10k+ predictions into DB each run).
4. Keep evaluation split frozen for meaningful comparisons.
5. Preserve immutable human labels; no silent overwrite.

## 4) Operating loop

Use this loop continuously:

1. Annotate from latest suggestions queue (`/annotate` queue mode; OCR shows latest model predictions).
2. Export + train (`run_automation.py` or `/jobs` automation button); policy auto-applied.
3. Run baseline/fine-tuned eval (`/jobs` eval controls; split-aware val/test submission).
4. Monitor results (`/jobs/results` for clear experiment trails).
5. Run full unlabeled inference (`/jobs` pool button).
6. Generate next suggestions (`/jobs` → Generate next samples button, or manual script).
7. Repeat.

## 5) Parallel work policy

While annotation is ongoing, parallel engineering should focus on:

- crop quality improvements without breaking ID matching,
- better candidate ranking quality,
- faster/safer operations (sync/export/infer),
- compact, up-to-date documentation.

Avoid broad UI expansion until loop quality is stable.

## 6) Active technical decisions

### 6.1 Configurable text-marker policy

- Marker handling (e.g. `-` prefixes, trailing metadata) is now configurable per profile.
- Policy applies automatically on export via `inkwell/text_policy.py`.
- Policy is persisted in dataset manifest for reproducibility.
- Keeps readable-text training clean while preserving full raw OCR in DB.

### 6.2 Crop quality

- Bottom margin is currently acceptable.
- Top margin should be slightly larger to reduce accent clipping.
- Safe approach:
  - recrop existing lines in place (prefer unannotated-only by default),
  - keep segmentation IDs/line IDs unchanged.

### 6.3 Full-pool inference storage

Store in job artifacts only:

- `jobs/infer_pool_.../pool_predictions.jsonl`
- `jobs/infer_pool_.../result.json`

Do not insert all predictions into DB.

### 6.4 Suggestions quality

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
