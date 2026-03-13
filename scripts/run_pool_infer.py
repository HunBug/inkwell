#!/usr/bin/env python3
"""
One-click launcher: export unlabeled pool + submit GPU infer_pool job.

Reads dataset/shared config from automation.toml.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "automation.toml"


class PoolInferError(RuntimeError):
    pass


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise PoolInferError(f"Missing config file: {CONFIG_PATH}")
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True, check=False)


def find_latest_finetune_checkpoint(shared: Path, dataset_id: str) -> str:
    jobs_dir = shared / "jobs"
    if not jobs_dir.exists():
        raise PoolInferError(f"Jobs dir not found: {jobs_dir}")

    best: tuple[str, Path] | None = None
    for d in jobs_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("finetune_"):
            continue
        job_file = d / "job.json"
        progress_file = d / "progress.json"
        checkpoint = d / "checkpoints" / "best"
        if not job_file.exists() or not progress_file.exists() or not checkpoint.exists():
            continue
        try:
            job = json.loads(job_file.read_text())
            progress = json.loads(progress_file.read_text())
        except Exception:
            continue
        if job.get("dataset_id") != dataset_id:
            continue
        if progress.get("status") != "completed":
            continue
        created_at = job.get("created_at", "")
        if best is None or created_at > best[0]:
            best = (created_at, checkpoint)

    if best is None:
        raise PoolInferError(
            f"No completed finetune checkpoint found for dataset {dataset_id}."
        )
    return str(best[1])


def main() -> None:
    cfg = load_config()
    dataset_id = cfg.get("dataset", {}).get("dataset_id")
    shared_path = cfg.get("shared", {}).get("path")
    if not dataset_id:
        raise PoolInferError("automation.toml [dataset].dataset_id is required")
    if not shared_path:
        raise PoolInferError("automation.toml [shared].path is required")

    shared = Path(shared_path).expanduser().resolve()
    if not shared.exists():
        raise PoolInferError(f"Shared path does not exist: {shared}")

    print(f"Dataset: {dataset_id}")
    print(f"Shared: {shared}")

    checkpoint = find_latest_finetune_checkpoint(shared, dataset_id)
    print(f"Using checkpoint: {checkpoint}")

    print("\n==> Export unlabeled pool")
    export_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "export_unlabeled_pool.py"),
        "--dataset-id", dataset_id,
        "--shared", str(shared),
    ]
    exp = run_cmd(export_cmd)
    if exp.returncode != 0:
        raise PoolInferError(exp.stderr or exp.stdout)
    print(exp.stdout.strip())

    print("\n==> Submit infer_pool job")
    submit_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "submit_job.py"),
        "infer_pool",
        "--dataset-id", dataset_id,
        "--checkpoint", checkpoint,
        "--shared", str(shared),
        "--label", f"Infer full pool {dataset_id}",
    ]
    sub = run_cmd(submit_cmd)
    if sub.returncode != 0:
        raise PoolInferError(sub.stderr or sub.stdout)
    print(sub.stdout.strip())

    print("\nDone: export completed and infer_pool job submitted.")


if __name__ == "__main__":
    try:
        main()
    except PoolInferError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
