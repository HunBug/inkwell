#!/usr/bin/env python3
"""
Smoke test: verifies the full hu_pretrain pipeline end-to-end.

Downloads 100 samples, runs 1 training epoch, checks that:
  - MLflow run was created and contains expected metrics
  - Checkpoint was saved and is loadable
  - Final val CER was logged

Intended to be run on the GPU node after any significant change.
Fast: ~2-5 minutes on RTX 2060.

Usage:
    python hu_pretrain/test_pipeline.py
    python hu_pretrain/test_pipeline.py --keep-output   # don't delete tmp dir
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_SCRIPT = REPO_ROOT / "hu_pretrain" / "download_data.py"
PRETRAIN_SCRIPT = REPO_ROOT / "hu_pretrain" / "pretrain.py"


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"[test] {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(f"[FAIL] {label} exited with code {result.returncode}")
    print(f"[OK] {label}")


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"[OK]   {message}")
    else:
        print(f"[FAIL] {message}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for hu_pretrain pipeline")
    parser.add_argument("--keep-output", action="store_true", help="Keep tmp dir after test")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="inkwell_hu_test_"))
    print(f"Temp dir: {tmp}")

    dataset_dir = tmp / "synth_hu_smoke"
    output_dir = tmp / "pretrain_out"
    tracking_uri = tmp / "mlflow"

    try:
        # --- Step 1: Download 100 samples ---
        run(
            [
                sys.executable, str(DOWNLOAD_SCRIPT),
                "--output", str(dataset_dir),
                "--count", "100",
                "--val-ratio", "0.1",
            ],
            "download_data: 100 samples",
        )

        check((dataset_dir / "train.jsonl").exists(), "train.jsonl exists")
        check((dataset_dir / "val.jsonl").exists(), "val.jsonl exists")
        check((dataset_dir / "manifest.json").exists(), "manifest.json exists")

        manifest = json.loads((dataset_dir / "manifest.json").read_text())
        check(manifest["total"] >= 90, f"at least 90 samples collected (got {manifest['total']})")
        check(manifest["train"] > manifest["val"], "train > val")

        crops = list((dataset_dir / "crops").glob("*.jpg"))
        check(len(crops) >= 90, f"at least 90 crop images (got {len(crops)})")

        # --- Step 2: Train 1 epoch ---
        run(
            [
                sys.executable, str(PRETRAIN_SCRIPT),
                "--dataset", str(dataset_dir),
                "--output", str(output_dir),
                "--epochs", "1",
                "--batch-size", "2",
                "--max-new-tokens", "32",
                "--tracking-uri", str(tracking_uri),
                "--run-name", "smoke_test",
            ],
            "pretrain: 1 epoch",
        )

        # --- Step 3: Check checkpoint ---
        best_dir = output_dir / "checkpoints" / "best"
        check(best_dir.exists(), "checkpoint/best dir exists")
        check((best_dir / "config.json").exists(), "config.json in checkpoint")
        check((best_dir / "tokenizer_config.json").exists() or
              (best_dir / "vocab.json").exists(), "processor saved in checkpoint")

        result_path = output_dir / "result.json"
        check(result_path.exists(), "result.json exists")
        result = json.loads(result_path.read_text())
        check("final_val_cer" in result, f"final_val_cer in result.json (got {result.get('final_val_cer')})")
        check(0.0 <= result["final_val_cer"] <= 1.0, f"val CER in [0, 1] (got {result['final_val_cer']:.4f})")

        # --- Step 4: Check MLflow ---
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri.as_uri())
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            check(len(experiments) >= 1, "MLflow experiment created")

            runs = client.search_runs([exp.experiment_id for exp in experiments])
            check(len(runs) >= 1, "at least one MLflow run")
            run_data = runs[0].data
            check("final_val_cer" in run_data.metrics, "final_val_cer logged in MLflow")
            check("train_loss_epoch" in run_data.metrics, "train_loss_epoch logged in MLflow")
        except ImportError:
            print("[SKIP] mlflow not installed; skipping MLflow checks")

        # --- Step 5: Verify checkpoint is loadable ---
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            proc = TrOCRProcessor.from_pretrained(str(best_dir), use_fast=True)
            mdl = VisionEncoderDecoderModel.from_pretrained(str(best_dir))
            check(mdl is not None, "checkpoint loads without error")
            del mdl, proc
        except Exception as exc:
            print(f"[FAIL] checkpoint load failed: {exc}")
            sys.exit(1)

        print("\n" + "="*60)
        print("[PASS] All checks passed.")
        print("="*60)

    finally:
        if args.keep_output:
            print(f"\nOutput kept at: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"\nTemp dir cleaned up.")


if __name__ == "__main__":
    main()
