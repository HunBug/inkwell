# Inkwell Session Handoff (for new LLM sessions)

## Current objective

Improve OCR quality through iterative human-in-the-loop training on handwritten Hungarian diaries.

## What is already working

- End-to-end pipeline (ingest → preprocess → segment → OCR → annotate → train/eval)
- GPU job queue and worker over shared folder
- `/jobs` controls for automation, sync, eval, and full-pool inference
- `/annotate` queue mode from suggestions files
- Full unlabeled pool export/inference path (`infer_pool`)

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

## Active quality issue

Line crops sometimes clip top accents; slight top-context increase is preferred over tight clipping.

Current mitigation:

- safe recrop tool: `scripts/recrop_lines.py`
- segmentation default adjusted to larger top margin for new segmentation runs

## Key scripts to know

- `scripts/run_automation.py`
- `scripts/sync_code_to_gpu.py`
- `scripts/export_unlabeled_pool.py`
- `scripts/infer_unlabeled_pool.py`
- `scripts/pick_next_samples.py`
- `scripts/recrop_lines.py`

## Next high-value tasks

1. Improve suggestion quality (reduce noisy/symbol-heavy picks).
2. Add mixed sampling buckets (hard + medium + cleaner lines).
3. Benchmark margin variants quickly on fixed sample and choose stable defaults.

## Read these docs first

1. `docs/inkwell_plan_final.md`
2. `docs/runbook.md`
3. `docs/session_handoff.md` (this file)

Historical docs are in `docs/archive/`.
