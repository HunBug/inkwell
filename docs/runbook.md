# Inkwell Runbook (Operational)

Minimal commands and flow for day-to-day work.

## 1) Environment

```bash
cd /home/akoss/Downloads/priv/code/inkwell
source .venv/bin/activate
```

## 2) Sync code to GPU worker (when code changed)

```bash
python scripts/sync_code_to_gpu.py --force --start-runner
```

## 3) Regular training/eval loop

1. Update `automation.toml` dataset id when starting a new milestone round.
2. Run automation:

```bash
python scripts/run_automation.py
```

3. Monitor in `/jobs`.
4. Submit baseline/fine-tuned eval jobs from `/jobs`.

## 4) Full unlabeled inference loop

Use when you want model predictions across the whole unlabeled pool.

### One-click (web)

- `/jobs` → **Export unlabeled + run full recognition**

### Manual equivalent

```bash
export SHARED=/home/akoss/mnt/lara-playground/playground/inkwell-automation
export DATASET_ID=gt_20260312_round1
CHECKPOINT=$(ls -dt "$SHARED"/jobs/finetune_*/checkpoints/best | head -n1)

python scripts/export_unlabeled_pool.py --dataset-id "$DATASET_ID" --shared "$SHARED" --force
python scripts/submit_job.py infer_pool --dataset-id "$DATASET_ID" --checkpoint "$CHECKPOINT" --shared "$SHARED" --infer-batch-size 16
```

## 5) Generate annotation suggestions

```bash
python scripts/pick_next_samples.py --n 150
```

Output:

- `working/suggestions/next_samples_*.jsonl`

Load in `/annotate` queue mode.

## 6) Safe crop-margin adjustment (no ID remap)

Increase top margin on existing crops while preserving line IDs.

```bash
python scripts/recrop_lines.py --top-extra 8 --bottom-extra 0 --only-unannotated
```

Dry run first:

```bash
python scripts/recrop_lines.py --top-extra 8 --bottom-extra 0 --dry-run
```

## 7) Where key artifacts live

- Shared queue/jobs: `.../inkwell-automation/jobs/`
- Shared datasets: `.../inkwell-automation/datasets/`
- Full-pool predictions: `jobs/infer_pool_.../pool_predictions.jsonl`
- Suggestions: `working/suggestions/next_samples_*.jsonl`
