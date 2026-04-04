#!/usr/bin/env python3
"""
Run full recognition on exported unlabeled_pool using a TrOCR checkpoint.

Inputs:
  --dataset  /shared/datasets/{dataset_id}
  --checkpoint  HF model id or local checkpoint path
  --job-dir /shared/jobs/infer_pool_... (optional)

Reads:
  dataset/unlabeled_pool/pool.jsonl

Writes:
  job-dir/pool_predictions.jsonl
  job-dir/result.json
  (and progress.json updates if --job-dir set)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def _generation_kwargs(max_new_tokens: int, num_beams: int) -> dict:
    return {
        "max_new_tokens": max(8, int(max_new_tokens)),
        "num_beams": max(1, int(num_beams)),
        "early_stopping": True,
    }


def write_progress(job_dir: Path | None, **kwargs) -> None:
    if job_dir is None:
        return
    p = job_dir / "progress.json"
    existing = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text())
        except Exception:
            pass
    existing.update(kwargs)
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    p.write_text(json.dumps(existing, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Run TrOCR inference on unlabeled pool")
    p.add_argument("--dataset", required=True, help="Shared dataset directory")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint (local path or HF ID)")
    p.add_argument("--job-dir", default=None, help="Job directory for progress/result files")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--num-beams", type=int, default=1)
    args = p.parse_args()

    dataset_dir = Path(args.dataset)
    job_dir = Path(args.job_dir) if args.job_dir else None

    pool_dir = dataset_dir / "unlabeled_pool"
    pool_jsonl = pool_dir / "pool.jsonl"
    if not pool_jsonl.exists():
        raise FileNotFoundError(
            f"Missing {pool_jsonl}. Run export_unlabeled_pool.py first."
        )

    write_progress(job_dir, status="running", message="Loading model...")
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)

    processor = TrOCRProcessor.from_pretrained(args.checkpoint, use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    generation_kwargs = _generation_kwargs(args.max_new_tokens, args.num_beams)

    items = []
    with open(pool_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    total = len(items)
    print(f"Running inference on {total} unlabeled lines", flush=True)
    write_progress(job_dir, status="running", total=total, done=0, message=f"Inferring {total} lines")

    out_path = (job_dir / "pool_predictions.jsonl") if job_dir else (pool_dir / "pool_predictions.jsonl")

    done = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for start in range(0, total, args.batch_size):
            batch = items[start:start + args.batch_size]
            images = []
            for item in batch:
                image_rel = Path(item["image"])
                image_path = pool_dir / image_rel
                if not image_path.exists():
                    # Backward-compatible fallback if image paths are ever stored
                    # relative to dataset root instead of unlabeled_pool root.
                    image_path = dataset_dir / image_rel
                images.append(Image.open(image_path).convert("RGB"))

            pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, **generation_kwargs)
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for item, pred in zip(batch, preds):
                record = {
                    "line_id": item["line_id"],
                    "page_id": item.get("page_id"),
                    "notebook_id": item.get("notebook_id"),
                    "notebook_folder": item.get("notebook_folder"),
                    "image": item.get("image"),
                    "ocr_auto_text": item.get("ocr_text") or "",
                    "ocr_auto_confidence": item.get("ocr_confidence"),
                    "predicted_text": (pred or "").strip(),
                    "checkpoint": args.checkpoint,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            done = min(start + args.batch_size, total)
            write_progress(job_dir, status="running", done=done, message=f"Inferring {done}/{total}")

    result = {
        "status": "completed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_dir),
        "checkpoint": args.checkpoint,
        "generation": generation_kwargs,
        "n_lines": total,
        "predictions_path": str(out_path),
    }

    if job_dir:
        (job_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        write_progress(job_dir, status="completed", message=f"Completed {total} lines", predictions_path=str(out_path))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
