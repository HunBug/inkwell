#!/usr/bin/env python3
"""
Evaluate a TrOCR checkpoint on the test (or val) split.

Reports CER and WER.  Called by gpu_worker.py, but can be run standalone.

Usage:
    python scripts/eval_model.py \\
        --dataset /shared/datasets/gt_20260311 \\
        --checkpoint /shared/jobs/finetune_.../checkpoints/best \\
        --job-dir /shared/jobs/eval_20260312_090000 \\
        --split test

    # Evaluate baseline (no fine-tune)
    python scripts/eval_model.py \\
        --dataset /shared/datasets/gt_20260311 \\
        --checkpoint microsoft/trocr-base-handwritten \\
        --split test
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ---------------------------------------------------------------------------
# CER / WER (no external deps)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            tmp = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def compute_cer(predictions: list[str], references: list[str]) -> float:
    total_dist = sum(_edit_distance(p, r) for p, r in zip(predictions, references))
    total_len = sum(len(r) for r in references)
    return total_dist / total_len if total_len > 0 else 0.0


def compute_wer(predictions: list[str], references: list[str]) -> float:
    total_dist = sum(
        _edit_distance(p.split(), r.split())
        for p, r in zip(predictions, references)
    )
    total_len = sum(len(r.split()) for r in references)
    return total_dist / total_len if total_len > 0 else 0.0


def write_progress(job_dir: Path | None, **kwargs) -> None:
    if job_dir is None:
        return
    p = job_dir / "progress.json"
    existing: dict = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text())
        except Exception:
            pass
    existing.update(kwargs)
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    p.write_text(json.dumps(existing, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TrOCR checkpoint")
    parser.add_argument("--dataset", required=True, help="Dataset directory")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (local path or HF model ID)")
    parser.add_argument("--job-dir", default=None, help="Job directory for progress/result files")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    job_dir = Path(args.job_dir) if args.job_dir else None

    write_progress(job_dir, status="running", message="Loading model...")
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)

    processor = TrOCRProcessor.from_pretrained(args.checkpoint, use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    jsonl_path = dataset_dir / f"{args.split}.jsonl"
    items: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    crops_dir = dataset_dir / "crops"
    print(f"Evaluating {len(items)} lines from split='{args.split}'...", flush=True)
    write_progress(job_dir, message=f"Evaluating {len(items)} lines...", total=len(items), done=0)

    predictions: list[str] = []
    references: list[str] = [item["text"] for item in items]
    per_line: list[dict] = []

    batch_pixel_values = []
    batch_items = []

    def flush_batch():
        if not batch_pixel_values:
            return
        tensor = torch.stack(batch_pixel_values).to(device)
        with torch.no_grad():
            generated_ids = model.generate(tensor)
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for item, pred in zip(batch_items, texts):
            pred = pred.strip()
            predictions.append(pred)
            per_line.append({
                "line_id": item.get("line_id"),
                "reference": item["text"],
                "prediction": pred,
                "cer": _edit_distance(pred, item["text"]) / max(len(item["text"]), 1),
            })
        batch_pixel_values.clear()
        batch_items.clear()

    for i, item in enumerate(tqdm(items, desc=f"Eval ({args.split})")):
        image_rel = Path(item["image"])
        if image_rel.parts and image_rel.parts[0] == "crops":
            image_path = crops_dir.parent / image_rel
        else:
            image_path = crops_dir / image_rel
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        batch_pixel_values.append(pixel_values)
        batch_items.append(item)
        if len(batch_pixel_values) >= args.batch_size:
            flush_batch()
            write_progress(job_dir, done=i + 1)
    flush_batch()

    cer = compute_cer(predictions, references)
    wer = compute_wer(predictions, references)

    print(f"\nResults on split='{args.split}':")
    print(f"  CER: {cer:.4f}  ({cer*100:.1f}%)")
    print(f"  WER: {wer:.4f}  ({wer*100:.1f}%)")
    print(f"  Lines: {len(items)}")

    # Show worst examples
    per_line.sort(key=lambda x: x["cer"], reverse=True)
    print("\nWorst 5 lines:")
    for ex in per_line[:5]:
        print(f"  CER={ex['cer']:.2f}  ref={ex['reference']!r}  pred={ex['prediction']!r}")

    result = {
        "status": "completed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_lines": len(items),
        "cer": cer,
        "wer": wer,
        "per_line": per_line,
    }

    if job_dir:
        (job_dir / "result.json").write_text(json.dumps(result, indent=2))
        write_progress(
            job_dir,
            status="completed",
            cer=cer,
            wer=wer,
            message=f"CER: {cer:.4f}  WER: {wer:.4f}",
        )
    else:
        # Print summary JSON to stdout
        summary = {k: v for k, v in result.items() if k != "per_line"}
        print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
