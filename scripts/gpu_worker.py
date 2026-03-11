#!/usr/bin/env python3
"""
GPU worker — runs on the GPU machine.

Polls the shared jobs folder for pending jobs and processes them one at a time.
Writes progress to progress.json.  Respects CANCEL file.  Writes result.json on completion.

Start it once (e.g. in a tmux session) on the GPU machine:

    export INKWELL_SHARED=/mnt/share/inkwell
    python scripts/gpu_worker.py

    # With a local copy of the dataset for performance:
    rsync -avz --progress $INKWELL_SHARED/datasets/ /local/inkwell/datasets/
    python scripts/gpu_worker.py --local-datasets /local/inkwell/datasets

The worker does NOT modify the shared folder for training — it reads datasets from
--local-datasets if provided (faster), and writes checkpoints + logs into the job folder.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

POLL_INTERVAL_SECS = 10


def write_worker_status(
    shared: Path,
    status: str,
    current_job: str | None = None,
    local_datasets: str | None = None,
    message: str | None = None,
) -> None:
    status_file = shared / "worker_status.json"
    payload = {
        "status": status,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "current_job": current_job,
        "local_datasets": local_datasets,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    status_file.write_text(json.dumps(payload, indent=2))


def get_shared_path(override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("INKWELL_SHARED")
    if env:
        return Path(env).expanduser().resolve()
    raise RuntimeError("Set INKWELL_SHARED env var or pass --shared")


def write_progress(job_dir: Path, **kwargs) -> None:
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


def is_cancelled(job_dir: Path) -> bool:
    return (job_dir / "CANCEL").exists()


def find_pending_job(shared: Path) -> Path | None:
    jobs_dir = shared / "jobs"
    if not jobs_dir.exists():
        return None
    candidates = []
    for d in sorted(jobs_dir.iterdir()):
        if not d.is_dir():
            continue
        job_file = d / "job.json"
        progress_file = d / "progress.json"
        if not job_file.exists() or not progress_file.exists():
            continue
        try:
            p = json.loads(progress_file.read_text())
        except Exception:
            continue
        if p.get("status") == "pending":
            created_at = json.loads(job_file.read_text()).get("created_at", "")
            candidates.append((created_at, d))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def run_job(job_dir: Path, local_datasets: Path | None) -> None:
    job = json.loads((job_dir / "job.json").read_text())
    job_type = job["type"]
    job_id = job["job_id"]

    log.info("Starting job %s  type=%s", job_id, job_type)
    write_worker_status(
        job_dir.parent.parent,
        status="running",
        current_job=job_id,
        local_datasets=str(local_datasets) if local_datasets else None,
        message=f"Running {job_type}",
    )
    write_progress(job_dir, status="running", message="Starting...")

    # Resolve dataset path: use local copy if available
    dataset_path = job["dataset_path"]
    if local_datasets is not None:
        local_ds = local_datasets / job["dataset_id"]
        if local_ds.exists():
            dataset_path = str(local_ds)
            log.info("Using local dataset copy: %s", dataset_path)
        else:
            log.warning("Local dataset not found (%s), using shared path", local_ds)

    scripts_dir = Path(__file__).parent
    log_file = job_dir / "worker.log"

    if job_type == "finetune":
        cmd = [
            sys.executable,
            str(scripts_dir / "finetune_trocr.py"),
            "--dataset", dataset_path,
            "--job-dir", str(job_dir),
            "--base-model", job["base_model"],
            "--epochs", str(job["params"]["epochs"]),
            "--batch-size", str(job["params"]["batch_size"]),
            "--lr", str(job["params"]["learning_rate"]),
        ]
        if job.get("resume_from"):
            cmd += ["--resume-from", job["resume_from"]]
    elif job_type == "eval":
        checkpoint = job.get("eval_checkpoint") or job.get("resume_from")
        if not checkpoint:
            raise ValueError("eval job requires eval_checkpoint in job.json")
        cmd = [
            sys.executable,
            str(scripts_dir / "eval_model.py"),
            "--dataset", dataset_path,
            "--checkpoint", checkpoint,
            "--job-dir", str(job_dir),
        ]
    else:
        raise ValueError(f"Unknown job type: {job_type}")

    log.info("Running: %s", " ".join(cmd))
    with open(log_file, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

        while proc.poll() is None:
            if is_cancelled(job_dir):
                log.warning("CANCEL file detected — terminating job %s", job_id)
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                write_progress(job_dir, status="cancelled", message="Cancelled by user")
                write_worker_status(
                    job_dir.parent.parent,
                    status="idle",
                    current_job=None,
                    local_datasets=str(local_datasets) if local_datasets else None,
                    message="Last job cancelled",
                )
                return
            time.sleep(5)
            write_worker_status(
                job_dir.parent.parent,
                status="running",
                current_job=job_id,
                local_datasets=str(local_datasets) if local_datasets else None,
                message=f"Running {job_type}",
            )

    rc = proc.returncode
    if rc == 0:
        # result.json is written by the subprocess scripts themselves
        # but if somehow it wasn't, write a minimal one
        result_file = job_dir / "result.json"
        if not result_file.exists():
            result_file.write_text(json.dumps({
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2))
        write_progress(job_dir, status="completed", message="Finished successfully")
        write_worker_status(
            job_dir.parent.parent,
            status="idle",
            current_job=None,
            local_datasets=str(local_datasets) if local_datasets else None,
            message="Idle",
        )
        log.info("Job %s completed successfully", job_id)
    else:
        write_progress(job_dir, status="failed", message=f"Process exited with code {rc}")
        write_worker_status(
            job_dir.parent.parent,
            status="idle",
            current_job=None,
            local_datasets=str(local_datasets) if local_datasets else None,
            message=f"Last job failed with exit code {rc}",
        )
        log.error("Job %s failed (exit code %d) — see %s", job_id, rc, log_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU worker — polls for and runs training jobs")
    parser.add_argument("--shared", default=None, help="Shared folder path (overrides INKWELL_SHARED)")
    parser.add_argument("--local-datasets", default=None, help="Local copy of datasets dir for faster I/O")
    parser.add_argument("--once", action="store_true", help="Process one pending job then exit")
    args = parser.parse_args()

    shared = get_shared_path(args.shared)
    local_datasets = Path(args.local_datasets).resolve() if args.local_datasets else None

    log.info("GPU worker started. Watching: %s", shared / "jobs")
    if local_datasets:
        log.info("Local dataset cache: %s", local_datasets)
    write_worker_status(
        shared,
        status="idle",
        current_job=None,
        local_datasets=str(local_datasets) if local_datasets else None,
        message="Worker started",
    )

    while True:
        write_worker_status(
            shared,
            status="idle",
            current_job=None,
            local_datasets=str(local_datasets) if local_datasets else None,
            message="Polling for jobs",
        )
        job_dir = find_pending_job(shared)
        if job_dir:
            try:
                run_job(job_dir, local_datasets)
            except Exception as exc:
                log.exception("Unexpected error running job in %s: %s", job_dir, exc)
                write_progress(job_dir, status="failed", message=str(exc))
                write_worker_status(
                    shared,
                    status="idle",
                    current_job=None,
                    local_datasets=str(local_datasets) if local_datasets else None,
                    message=f"Worker error: {exc}",
                )
            if args.once:
                break
        else:
            if args.once:
                log.info("No pending jobs found.")
                break
            time.sleep(POLL_INTERVAL_SECS)


if __name__ == "__main__":
    main()
