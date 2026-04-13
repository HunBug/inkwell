# hu_pretrain — Hungarian TrOCR pre-trainer

Pre-trains TrOCR on synthetic Hungarian handwriting data before fine-tuning on real diary annotations.

**Why:** `trocr-base-handwritten` has an English language prior (trained on IAM corpus). Pre-training on Hungarian synthetic images replaces this with Hungarian vocabulary, reducing English hallucinations on Hungarian text.

**Strategy:**
1. Download ~20k synthetic Hungarian line images from HuggingFace ([AlhitawiMohammed22/lines_hu_v2_1](https://huggingface.co/datasets/AlhitawiMohammed22/lines_hu_v2_1), Apache 2.0)
2. Pre-train `trocr-base-handwritten` on this data → Hungarian-biased checkpoint
3. Fine-tune that checkpoint on real diary annotations using the existing `scripts/finetune_trocr.py`

---

## Setup (GPU node)

```bash
# From inkwell repo root (already synced by sync_code_to_gpu.py)
pip install -r requirements-ml.txt -r hu_pretrain/requirements.txt
```

---

## Step 1: Download data

```bash
python hu_pretrain/download_data.py \
    --output working/shared/datasets/synth_hu_20k \
    --count 20000 \
    --val-ratio 0.05
```

Takes ~10-20 minutes over a typical connection. Output is in the same JSONL+crops format as existing GT datasets.

If you see an auth error, the dataset may require a HuggingFace login:
```bash
huggingface-cli login
```

---

## Step 2: Pre-train

```bash
python hu_pretrain/pretrain.py \
    --dataset working/shared/datasets/synth_hu_20k \
    --output working/shared/pretrained_hu \
    --epochs 3 \
    --batch-size 8
```

Estimated time on RTX 2060: ~2-4 hours for 20k × 3 epochs.

Checkpoint saved to: `working/shared/pretrained_hu/checkpoints/best/`

---

## Step 3: Fine-tune on real diary data

Use the existing automation flow, but point `--base-model` at the pre-trained checkpoint:

```bash
python scripts/finetune_trocr.py \
    --dataset working/shared/datasets/<latest_gt> \
    --job-dir working/shared/jobs/finetune_hu_pretrained \
    --base-model working/shared/pretrained_hu/checkpoints/best \
    --epochs 10 --batch-size 4
```

Or edit `automation.toml` to add a `base_model` override for this run.

---

## Tracking

MLflow logs are written to `working/shared/mlflow/` (on the NFS share, accessible from dev machine).

To view on dev machine:
```bash
mlflow ui --backend-store-uri /path/to/working/shared/mlflow
# open http://localhost:5000
```

---

## Smoke test (run after any change)

```bash
python hu_pretrain/test_pipeline.py
```

Downloads 100 samples, runs 1 epoch, checks checkpoint, checks MLflow. ~2-5 min.

```bash
python hu_pretrain/test_pipeline.py --keep-output   # inspect tmp files
```

---

## Files

| File | Purpose |
|------|---------|
| `download_data.py` | Stream from HuggingFace → JSONL+crops |
| `pretrain.py` | Training loop with MLflow tracking |
| `test_pipeline.py` | End-to-end smoke test |
| `requirements.txt` | `mlflow` + `datasets` |
