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
import signal
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
WORKER_SHARED: Path | None = None
WORKER_LOCAL_DATASETS: Path | None = None
WORKER_CURRENT_JOB: str | None = None
STOP_REQUESTED = False


def _sync_dataset_to_local_cache(shared: Path, dataset_id: str, local_datasets: Path) -> Path:
    """Rsync one dataset from shared storage to local cache and return local path."""
    src = shared / "datasets" / dataset_id
    dst = local_datasets / dataset_id

    if not src.exists():
        raise FileNotFoundError(f"Shared dataset path not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-az",
        "--delete",
        f"{str(src)}/",
        f"{str(dst)}/",
    ]
    res = subprocess.run(cmd, text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Dataset rsync failed for {dataset_id}: {res.stderr or res.stdout}"
        )
    return dst


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


def _handle_termination(signum, frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    if WORKER_SHARED is not None:
        try:
            write_worker_status(
                WORKER_SHARED,
                status="stopped",
                current_job=WORKER_CURRENT_JOB,
                local_datasets=str(WORKER_LOCAL_DATASETS) if WORKER_LOCAL_DATASETS else None,
                message=f"Worker stopping on signal {signum}",
            )
        except Exception:
            pass
    raise KeyboardInterrupt


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


def _resolve_shared_path_on_server(raw_path: str | None, shared: Path) -> str | None:
    """
    Translate a client-side shared absolute path to the server-side shared root.

    Example:
      client job.json may contain
        /home/akoss/.../inkwell-automation/datasets/gt_20260311
      while server sees the same shared folder as
        /home/hunbug/.../inkwell-automation

    In that case we remap the suffix under `datasets/` or `jobs/` onto the local
    server-side shared root.
    """
    if not raw_path:
        return None

    p = Path(raw_path).expanduser()
    if p.exists():
        return str(p.resolve())

    parts = list(p.parts)
    for anchor in ("datasets", "jobs"):
        if anchor in parts:
            idx = parts.index(anchor)
            remapped = shared / Path(*parts[idx:])
            return str(remapped)

    return str(p)


def run_job(job_dir: Path, local_datasets: Path | None, auto_sync_local_datasets: bool) -> None:
    global WORKER_CURRENT_JOB
    job = json.loads((job_dir / "job.json").read_text())
    job_type = job["type"]
    job_id = job["job_id"]
    WORKER_CURRENT_JOB = job_id

    log.info("Starting job %s  type=%s", job_id, job_type)
    write_worker_status(
        job_dir.parent.parent,
        status="running",
        current_job=job_id,
        local_datasets=str(local_datasets) if local_datasets else None,
        message=f"Running {job_type}",
    )
    write_progress(job_dir, status="running", message="Starting...")

    shared_root = job_dir.parent.parent

    # Resolve dataset path from server-side shared root, not client absolute path.
    dataset_path = str(shared_root / "datasets" / job["dataset_id"])

    preferred_dataset_path = _resolve_shared_path_on_server(
        job.get("preferred_dataset_path"),
        shared_root,
    )
    if preferred_dataset_path and Path(preferred_dataset_path).exists():
        dataset_path = preferred_dataset_path
        log.info("Using preferred local dataset path: %s", dataset_path)

    if local_datasets is not None:
        if auto_sync_local_datasets:
            try:
                write_progress(
                    job_dir,
                    status="running",
                    message=f"Syncing dataset to local cache: {local_datasets / job['dataset_id']}",
                )
                synced = _sync_dataset_to_local_cache(shared_root, job["dataset_id"], local_datasets)
                dataset_path = str(synced)
                log.info("Synced dataset to local cache: %s", dataset_path)
                write_progress(
                    job_dir,
                    status="running",
                    message=f"Dataset synced to local cache: {dataset_path}",
                    dataset_path=dataset_path,
                )
            except Exception as exc:
                log.warning("Local dataset auto-sync failed, falling back: %s", exc)
                write_progress(
                    job_dir,
                    status="running",
                    message=f"Local dataset auto-sync failed, falling back to shared path: {exc}",
                )

        local_ds = local_datasets / job["dataset_id"]
        if local_ds.exists():
            dataset_path = str(local_ds)
            log.info("Using local dataset copy: %s", dataset_path)
            write_progress(
                job_dir,
                status="running",
                message=f"Using local dataset copy: {dataset_path}",
                dataset_path=dataset_path,
            )
        else:
            log.warning("Local dataset not found (%s), using shared path", local_ds)
            write_progress(
                job_dir,
                status="running",
                message=f"Local dataset cache missing, using shared path: {dataset_path}",
                dataset_path=dataset_path,
            )

    if not Path(dataset_path).exists():
        # Final fallback for old jobs that stored client-side absolute dataset_path.
        remapped = _resolve_shared_path_on_server(job.get("dataset_path"), shared_root)
        if remapped and Path(remapped).exists():
            dataset_path = remapped
            log.info("Remapped dataset path to server shared root: %s", dataset_path)
        else:
            raise FileNotFoundError(
                f"Dataset path not found on server: {dataset_path}"
            )

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
            resume_from = _resolve_shared_path_on_server(job["resume_from"], shared_root)
            cmd += ["--resume-from", resume_from]
    elif job_type == "eval":
        checkpoint = job.get("eval_checkpoint") or job.get("resume_from")
        if not checkpoint:
            raise ValueError("eval job requires eval_checkpoint in job.json")
        checkpoint = _resolve_shared_path_on_server(checkpoint, shared_root)
        cmd = [
            sys.executable,
            str(scripts_dir / "eval_model.py"),
            "--dataset", dataset_path,
            "--checkpoint", checkpoint,
            "--job-dir", str(job_dir),
            "--split", job.get("split", "val"),
        ]
    elif job_type == "infer_pool":
        checkpoint = job.get("eval_checkpoint") or job.get("resume_from")
        if not checkpoint:
            raise ValueError("infer_pool job requires eval_checkpoint in job.json")
        checkpoint = _resolve_shared_path_on_server(checkpoint, shared_root)
        infer_batch_size = str(job.get("params", {}).get("infer_batch_size", 16))
        cmd = [
            sys.executable,
            str(scripts_dir / "infer_unlabeled_pool.py"),
            "--dataset", dataset_path,
            "--checkpoint", checkpoint,
            "--job-dir", str(job_dir),
            "--batch-size", infer_batch_size,
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
                WORKER_CURRENT_JOB = None
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
        WORKER_CURRENT_JOB = None
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
        WORKER_CURRENT_JOB = None
        log.error("Job %s failed (exit code %d) — see %s", job_id, rc, log_file)


def main() -> None:
    global WORKER_SHARED, WORKER_LOCAL_DATASETS
    parser = argparse.ArgumentParser(description="GPU worker — polls for and runs training jobs")
    parser.add_argument("--shared", default=None, help="Shared folder path (overrides INKWELL_SHARED)")
    parser.add_argument("--local-datasets", default=None, help="Local copy of datasets dir for faster I/O")
    parser.add_argument(
        "--no-auto-sync-local-datasets",
        action="store_true",
        help="Disable automatic rsync of each dataset into --local-datasets before running a job",
    )
    parser.add_argument("--once", action="store_true", help="Process one pending job then exit")
    args = parser.parse_args()

    shared = get_shared_path(args.shared)
    local_datasets = Path(args.local_datasets).resolve() if args.local_datasets else None
    auto_sync_local_datasets = not args.no_auto_sync_local_datasets
    WORKER_SHARED = shared
    WORKER_LOCAL_DATASETS = local_datasets

    signal.signal(signal.SIGTERM, _handle_termination)
    signal.signal(signal.SIGINT, _handle_termination)

    log.info("GPU worker started. Watching: %s", shared / "jobs")
    if local_datasets:
        log.info("Local dataset cache: %s", local_datasets)
        log.info("Auto-sync local datasets: %s", auto_sync_local_datasets)
    write_worker_status(
        shared,
        status="idle",
        current_job=None,
        local_datasets=str(local_datasets) if local_datasets else None,
        message="Worker started",
    )

    while True:
        if STOP_REQUESTED:
            break
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
                run_job(job_dir, local_datasets, auto_sync_local_datasets)
                
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

    write_worker_status(
        shared,
        status="stopped",
        current_job=None,
        local_datasets=str(local_datasets) if local_datasets else None,
        message="Worker exited",
    )


if __name__ == "__main__":
    main()
