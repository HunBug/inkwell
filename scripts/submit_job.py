#!/usr/bin/env python3
"""
Submit a finetune or eval job to the shared job queue.

Creates:
  SHARED/jobs/{job_id}/
    job.json
    progress.json   (status: pending)

The GPU worker picks this up automatically.

Usage:
    # Finetune from base model
    python scripts/submit_job.py finetune --dataset-id gt_20260311

    # Finetune continuing from a previous checkpoint
    python scripts/submit_job.py finetune --dataset-id gt_20260311 \\
        --resume-from /shared/inkwell/jobs/finetune_.../checkpoints/best

    # Evaluate a checkpoint against the test set
    python scripts/submit_job.py eval --dataset-id gt_20260311 \\
        --checkpoint /shared/inkwell/jobs/finetune_.../checkpoints/best

    # Custom parameters
    python scripts/submit_job.py finetune --dataset-id gt_20260311 \\
        --base-model microsoft/trocr-large-handwritten \\
        --epochs 20 --batch-size 4 --lr 2e-5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_shared_path(override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("INKWELL_SHARED")
    if env:
        return Path(env).expanduser().resolve()
    fallback = PROJECT_ROOT / "working" / "shared"
    print(f"[warn] INKWELL_SHARED not set, using fallback: {fallback}")
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a GPU job")
    parser.add_argument("type", choices=["finetune", "eval"], help="Job type")
    parser.add_argument("--dataset-id", required=True, help="Dataset ID (from export_gt.py)")
    parser.add_argument("--shared", default=None, help="Shared folder path")
    parser.add_argument(
        "--base-model",
        default="microsoft/trocr-base-handwritten",
        help="Base HuggingFace checkpoint",
    )
    parser.add_argument("--resume-from", default=None, help="Continue finetune from this checkpoint path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for eval jobs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--label", default=None, help="Optional human label for this job")
    args = parser.parse_args()

    shared = get_shared_path(args.shared)
    now = datetime.now(timezone.utc)
    job_id = f"{args.type}_{now.strftime('%Y%m%d_%H%M%S')}"
    job_dir = shared / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = str(shared / "datasets" / args.dataset_id)

    job = {
        "job_id": job_id,
        "type": args.type,
        "label": args.label or job_id,
        "created_at": now.isoformat(),
        "dataset_id": args.dataset_id,
        "dataset_path": dataset_path,
        "base_model": args.base_model,
        "resume_from": args.resume_from,
        "eval_checkpoint": args.checkpoint,
        "params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        },
    }
    with open(job_dir / "job.json", "w") as f:
        json.dump(job, f, indent=2)

    progress = {
        "status": "pending",
        "message": "Waiting for GPU worker",
        "updated_at": now.isoformat(),
    }
    with open(job_dir / "progress.json", "w") as f:
        json.dump(progress, f, indent=2)

    print(f"Job submitted: {job_id}")
    print(f"  Type:       {args.type}")
    print(f"  Dataset:    {args.dataset_id}")
    if args.type == "finetune":
        print(f"  Base model: {args.base_model}")
        print(f"  Epochs:     {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}")
        if args.resume_from:
            print(f"  Resume from: {args.resume_from}")
    else:
        print(f"  Checkpoint: {args.checkpoint}")
    print(f"\nJob directory: {job_dir}")
    print("To cancel:  touch " + str(job_dir / "CANCEL"))


if __name__ == "__main__":
    main()
