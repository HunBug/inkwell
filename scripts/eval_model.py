#!/usr/bin/env python3
"""
Evaluate a TrOCR checkpoint on the test (or val) split.

Reports CER and WER.  Called by gpu_worker.py, but can also be run standalone
from the dev machine to evaluate baseline and compare models.

Usage:
    # Evaluate a checkpoint (GPU worker mode, writes progress/result.json):
    python scripts/eval_model.py \\
        --dataset /shared/datasets/gt_20260311 \\
        --checkpoint /shared/jobs/finetune_.../checkpoints/best \\
        --job-dir /shared/jobs/eval_20260312_090000 \\
        --split test

    # Evaluate baseline (standalone, saves to DB + working/evals/):
    python scripts/eval_model.py \\
        --dataset /path/to/datasets/gt_20260312_round1 \\
        --checkpoint microsoft/trocr-base-handwritten \\
        --split val

    # Compare baseline vs fine-tuned job side-by-side:
    python scripts/eval_model.py \\
        --compare-job finetune_20260312_140456 \\
        --dataset /path/to/datasets/gt_20260312_round1 \\
        --split val
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

import torch
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from inkwell.db import get_connection, DEFAULT_DB_PATH


# ---------------------------------------------------------------------------
# DB helpers  (eval_runs table — created on demand, not via init_db.py)
# ---------------------------------------------------------------------------

_EVAL_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id                INTEGER PRIMARY KEY,
    eval_id           TEXT NOT NULL UNIQUE,
    model_name        TEXT NOT NULL,
    model_path        TEXT,
    dataset_id        TEXT NOT NULL,
    split             TEXT NOT NULL DEFAULT 'val',
    num_samples       INTEGER,
    cer               REAL,
    wer               REAL,
    predictions_path  TEXT,
    created_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


def _ensure_eval_runs_table(conn) -> None:
    conn.execute(_EVAL_RUNS_DDL)
    conn.commit()


def _save_eval_run(conn, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO eval_runs
            (eval_id, model_name, model_path, dataset_id, split,
             num_samples, cer, wer, predictions_path, created_at)
        VALUES
            (:eval_id, :model_name, :model_path, :dataset_id, :split,
             :num_samples, :cer, :wer, :predictions_path, :created_at)
        """,
        row,
    )
    conn.commit()


def _model_slug(checkpoint: str) -> str:
    """Short filesystem-safe label for an eval_id."""
    p = Path(checkpoint)
    if p.exists():
        for part in reversed(p.parts):
            if part.startswith("finetune_"):
                return part
        return p.parent.name or p.name
    return p.name.replace("_", "-")


def _resolve_shared(shared_arg: str | None) -> Path:
    if shared_arg:
        return Path(shared_arg).expanduser().resolve()
    env = os.environ.get("INKWELL_SHARED")
    if env:
        return Path(env).expanduser().resolve()
    return Path("/home/akoss/mnt/lara-playground/playground/inkwell-automation")


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


def _generation_kwargs(max_new_tokens: int, num_beams: int) -> dict:
    return {
        "max_new_tokens": max(8, int(max_new_tokens)),
        "num_beams": max(1, int(num_beams)),
        "early_stopping": True,
    }


def _run_eval(
    checkpoint: str,
    dataset_dir: Path,
    split: str,
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
    job_dir: Path | None,
    output_dir: Path | None,
    db_path: Path | None,
) -> dict:
    """Core eval logic. Returns a result dict with CER/WER and per_line list."""

    write_progress(job_dir, status="running", message="Loading model...")
    print(f"Loading checkpoint: {checkpoint}", flush=True)

    processor = TrOCRProcessor.from_pretrained(checkpoint, use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    jsonl_path = dataset_dir / f"{split}.jsonl"
    items: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    crops_dir = dataset_dir / "crops"
    print(f"Evaluating {len(items)} lines from split='{split}'...", flush=True)
    write_progress(job_dir, message=f"Evaluating {len(items)} lines...", total=len(items), done=0)

    predictions: list[str] = []
    references: list[str] = [item["text"] for item in items]
    per_line: list[dict] = []
    generation_kwargs = _generation_kwargs(max_new_tokens=max_new_tokens, num_beams=num_beams)

    batch_pixel_values = []
    batch_items = []

    def flush_batch():
        if not batch_pixel_values:
            return
        tensor = torch.stack(batch_pixel_values).to(device)
        with torch.no_grad():
            generated_ids = model.generate(tensor, **generation_kwargs)
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for item, pred in zip(batch_items, texts):
            pred = pred.strip()
            predictions.append(pred)
            per_line.append({
                "line_id": item.get("line_id"),
                "image": item.get("image"),
                "reference": item["text"],
                "prediction": pred,
                "cer": round(_edit_distance(pred, item["text"]) / max(len(item["text"]), 1), 4),
            })
        batch_pixel_values.clear()
        batch_items.clear()

    for i, item in enumerate(tqdm(items, desc=f"Eval ({split})")):
        image_rel = Path(item["image"])
        if image_rel.parts and image_rel.parts[0] == "crops":
            image_path = crops_dir.parent / image_rel
        else:
            image_path = crops_dir / image_rel
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        batch_pixel_values.append(pixel_values)
        batch_items.append(item)
        if len(batch_pixel_values) >= batch_size:
            flush_batch()
            write_progress(job_dir, done=i + 1)
    flush_batch()

    cer = compute_cer(predictions, references)
    wer = compute_wer(predictions, references)

    print(f"\nResults on split='{split}':")
    print(f"  CER: {cer:.4f}  ({cer*100:.1f}%)")
    print(f"  WER: {wer:.4f}  ({wer*100:.1f}%)")
    print(f"  Lines: {len(items)}")

    per_line_sorted = sorted(per_line, key=lambda x: x["cer"], reverse=True)
    print("\nWorst 5 lines:")
    for ex in per_line_sorted[:5]:
        print(f"  CER={ex['cer']:.2f}  ref={ex['reference']!r}  pred={ex['prediction']!r}")

    result = {
        "status": "completed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "split": split,
        "generation": generation_kwargs,
        "n_lines": len(items),
        "cer": cer,
        "wer": wer,
        "per_line": per_line,
    }

    # --- GPU worker mode: write progress/result.json into job_dir ---
    if job_dir:
        (job_dir / "result.json").write_text(json.dumps(result, indent=2))
        write_progress(
            job_dir,
            status="completed",
            cer=cer,
            wer=wer,
            message=f"CER: {cer:.4f}  WER: {wer:.4f}",
        )

    # --- Standalone mode: save to working/evals/ + DB ---
    dataset_id = dataset_dir.name
    slug = _model_slug(checkpoint)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    eval_id = f"eval_{slug}_{dataset_id}_{split}_{ts}"

    out_base = output_dir or (PROJECT_ROOT / "working" / "evals")
    eval_dir = out_base / eval_id
    eval_dir.mkdir(parents=True, exist_ok=True)

    preds_path = eval_dir / "predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for row in per_line:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "eval_id": eval_id,
        "model_name": checkpoint,
        "model_path": str(checkpoint) if Path(checkpoint).exists() else None,
        "dataset_id": dataset_id,
        "split": split,
        "num_samples": len(per_line),
        "cer": cer,
        "wer": wer,
        "predictions_path": str(preds_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(eval_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    conn = get_connection(db_path)
    _ensure_eval_runs_table(conn)
    _save_eval_run(conn, summary)
    conn.close()
    print(f"[eval] Saved → {eval_dir}")

    result["eval_id"] = eval_id
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TrOCR checkpoint")
    parser.add_argument("--dataset", required=True, help="Dataset directory")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint (local path or HF model ID). Required unless --compare-job is used.",
    )
    parser.add_argument("--job-dir", default=None, help="Job directory for progress/result files (GPU worker mode)")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    # Standalone extras
    parser.add_argument(
        "--compare-job",
        metavar="JOB_ID",
        default=None,
        help="Evaluate BOTH baseline AND this fine-tune job (e.g. finetune_20260312_140456)",
    )
    parser.add_argument(
        "--baseline-model",
        default="microsoft/trocr-base-handwritten",
        help="Baseline model for --compare-job mode (default: microsoft/trocr-base-handwritten)",
    )
    parser.add_argument(
        "--shared",
        default=None,
        help="Shared folder root (for resolving --compare-job checkpoint path)",
    )
    parser.add_argument("--output-dir", default=None, help="Override output dir for eval results (default: working/evals/)")
    parser.add_argument("--db", default=None, help="Override DB path")
    args = parser.parse_args()

    if not args.checkpoint and not args.compare_job:
        parser.error("Either --checkpoint or --compare-job is required")

    dataset_dir = Path(args.dataset).expanduser().resolve()
    job_dir = Path(args.job_dir) if args.job_dir else None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    db_path = Path(args.db).expanduser().resolve() if args.db else None

    common = dict(
        dataset_dir=dataset_dir,
        split=args.split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        job_dir=job_dir,
        output_dir=output_dir,
        db_path=db_path,
    )

    if args.compare_job:
        shared = _resolve_shared(args.shared)
        cp_path = shared / "jobs" / args.compare_job / "checkpoints" / "best"
        if not cp_path.exists():
            parser.error(f"Checkpoint not found: {cp_path}")

        print("=" * 60)
        print(f"BASELINE: {args.baseline_model}")
        print("=" * 60)
        baseline = _run_eval(checkpoint=args.baseline_model, **common)

        print("\n" + "=" * 60)
        print(f"FINE-TUNED: {cp_path}")
        print("=" * 60)
        finetuned = _run_eval(checkpoint=str(cp_path), **common)

        delta_cer = finetuned["cer"] - baseline["cer"]
        delta_wer = finetuned["wer"] - baseline["wer"]
        sign_cer = "↓" if delta_cer < 0 else "↑"
        sign_wer = "↓" if delta_wer < 0 else "↑"
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print(f"  CER: {baseline['cer']:.4f} → {finetuned['cer']:.4f}  {sign_cer} {abs(delta_cer)*100:.1f}pp")
        print(f"  WER: {baseline['wer']:.4f} → {finetuned['wer']:.4f}  {sign_wer} {abs(delta_wer)*100:.1f}pp")
        print("=" * 60)
    else:
        _run_eval(checkpoint=args.checkpoint, **common)


if __name__ == "__main__":
    main()
