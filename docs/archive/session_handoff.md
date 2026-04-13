# Inkwell Session Handoff (for new LLM sessions)

## Current objective

Improve OCR quality through iterative human-in-the-loop training on handwritten Hungarian diaries.

## What is already working

- End-to-end pipeline (ingest → preprocess → segment → OCR → annotate → train/eval)
- GPU job queue and worker over shared folder
- `/jobs` controls for automation, sync, eval, full-pool inference, and suggestions generation (one-click)
- `/jobs/results` for clean experiment + eval overview
- `/annotate` queue mode from suggestions files; OCR preview from latest model predictions
- Full unlabeled pool export/inference path (`infer_pool`)
- Configurable text-marker policy for readable training data export

## Core loop

1. annotate suggested lines,
2. run automation,
3. evaluate baseline vs fine-tuned,
4. run full-pool inference,
5. regenerate suggestions,
6. repeat.

## Important conventions

- `dataset_id` is an iteration namespace (e.g. `gt_20260312_round1`), not corpus size.
- Full-pool predictions remain file-based (no large DB ingestion).
- Preserve existing line IDs and immutable human labels.

## Quality notes

**Line crops:** sometimes clip top accents; slight top-context increase preferred over tight clipping.
- Safe recrop tool: `scripts/recrop_lines.py`
- Segmentation default adjusted to larger top margin for new runs

**Text rendering:** Configurable marker policy (e.g., `-` prefixes) keeps training data readable.
- Controlled via `automation.toml` policies
- Auto-applied on export; results tracked in manifest

## Key scripts to know

- `scripts/run_automation.py`
- `scripts/sync_code_to_gpu.py`
- `scripts/export_unlabeled_pool.py`
- `scripts/infer_unlabeled_pool.py`
- `scripts/pick_next_samples.py`
- `scripts/recrop_lines.py`

## Latest status (end of session)

- Round 3 complete with verified fine-tuned checkpoint and full-pool inference
- Fresh suggestions file generated and confirmed loadable
- One-click "Generate next samples" button added to `/jobs` page
- Marker policy system integrated and tested in export pipeline
- Results page deployed for clearer eval/experiment tracking

## Next high-value tasks

1. Continue annotation on latest suggestions (queue-mode in `/annotate`).
2. Monitor fine-tune/eval stability over successive rounds.
3. Benchmark crop margin variants on next round cycle.

## Read these docs first

1. `docs/inkwell_plan_final.md`
2. `docs/runbook.md`
3. `docs/session_handoff.md` (this file)

Historical docs are in `docs/archive/`.
