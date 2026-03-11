#!/usr/bin/env python3
"""
One-command launcher for the TRoCR train/eval loop.

Reads configuration from ./automation.toml and performs:
  1. prechecks (shared path, GPU ping, worker heartbeat, active jobs)
  2. incremental split assignment
  3. GT export if needed (or dataset reuse if already exported)
  4. optional rsync of dataset to GPU local cache
  5. job submission via shared-folder queue

Usage:
    python scripts/run_automation.py
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inkwell.db import DEFAULT_DB_PATH, get_connection

CONFIG_PATH = PROJECT_ROOT / "automation.toml"
ACTIVE_STATUSES = {"pending", "running"}


class AutomationError(RuntimeError):
    pass


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise AutomationError(f"Missing config file: {CONFIG_PATH}")
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )


def print_step(message: str) -> None:
    print(f"\n==> {message}")


def get_shared_path(cfg: dict) -> Path:
    shared_path = cfg.get("shared", {}).get("path")
    if not shared_path:
        raise AutomationError("automation.toml: [shared].path is required")
    return Path(shared_path).expanduser().resolve()


def precheck_shared_path(shared: Path) -> None:
    """
    Validate that the shared mount/base path is accessible before any writes.

    Important safety behavior:
    - We do NOT create parent directories automatically.
    - This avoids silently writing to a local path when a network mount is missing.
    """
    base_path = shared.parent

    if not base_path.exists():
        raise AutomationError(
            "Shared base path does not exist (possible missing mount): "
            f"{base_path}"
        )

    if not os.access(base_path, os.R_OK | os.X_OK):
        raise AutomationError(
            "Shared base path is not readable/executable: "
            f"{base_path}"
        )

    if not shared.exists():
        if not os.access(base_path, os.W_OK | os.X_OK):
            raise AutomationError(
                "Cannot create shared path under base path (no write permission): "
                f"{base_path}"
            )
        shared.mkdir(exist_ok=True)

    if not shared.is_dir():
        raise AutomationError(f"Shared path is not a directory: {shared}")

    if not os.access(shared, os.R_OK | os.W_OK | os.X_OK):
        raise AutomationError(
            "Shared path lacks read/write/execute permissions: "
            f"{shared}"
        )

    # Lightweight write test
    try:
        with tempfile.NamedTemporaryFile(prefix=".inkwell_precheck_", dir=shared, delete=True):
            pass
    except Exception as exc:
        raise AutomationError(f"Shared path write test failed for {shared}: {exc}") from exc

    print(f"Shared path OK: {shared}")
    print(f"Shared base path OK: {base_path}")


def get_ssh_target(cfg: dict) -> str:
    gpu = cfg.get("gpu", {})
    host = gpu.get("host")
    if not host:
        raise AutomationError("automation.toml: [gpu].host is required")
    user = gpu.get("user")
    return f"{user}@{host}" if user else str(host)


def ping_gpu(cfg: dict) -> None:
    host = cfg.get("gpu", {}).get("host")
    if not host:
        raise AutomationError("GPU host is not configured")
    print(f"Pinging {host}...")
    result = run_cmd(["ping", "-c", "1", "-W", "2", host])
    if result.returncode != 0:
        raise AutomationError(f"GPU host is not reachable: {host}\n{result.stderr or result.stdout}")
    print("Ping OK")


def load_worker_status(shared: Path) -> dict | None:
    status_file = shared / "worker_status.json"
    if not status_file.exists():
        return None
    try:
        return json.loads(status_file.read_text())
    except Exception:
        return None


def heartbeat_age_seconds(worker_status: dict) -> float | None:
    updated_at = worker_status.get("updated_at")
    if not updated_at:
        return None
    try:
        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except Exception:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds()


def active_jobs(shared: Path) -> list[dict]:
    jobs_dir = shared / "jobs"
    if not jobs_dir.exists():
        return []
    active: list[dict] = []
    for job_dir in sorted(jobs_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        progress_file = job_dir / "progress.json"
        job_file = job_dir / "job.json"
        if not progress_file.exists() or not job_file.exists():
            continue
        try:
            progress = json.loads(progress_file.read_text())
            job = json.loads(job_file.read_text())
        except Exception:
            continue
        if progress.get("status") in ACTIVE_STATUSES:
            active.append({
                "job_id": job.get("job_id", job_dir.name),
                "status": progress.get("status"),
                "message": progress.get("message", ""),
            })
    return active


def count_current_gt() -> int:
    conn = get_connection(str(DEFAULT_DB_PATH))
    row = conn.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM transcriptions t
        WHERE t.transcription_type = 'HUMAN_CORRECTED'
          AND t.immutable = 1
          AND t.text NOT IN ('[ur]', '[nt]', '[?]')
        """
    ).fetchone()
    return int(row["cnt"])


def assign_splits() -> None:
    result = run_cmd([sys.executable, str(PROJECT_ROOT / "scripts" / "assign_splits.py")], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise AutomationError(result.stderr or result.stdout)
    print(result.stdout.strip())


def ensure_dataset(cfg: dict, shared: Path) -> Path:
    dataset_cfg = cfg.get("dataset", {})
    dataset_id = dataset_cfg.get("dataset_id")
    if not dataset_id:
        raise AutomationError("automation.toml: [dataset].dataset_id is required")

    dataset_dir = shared / "datasets" / dataset_id
    manifest_file = dataset_dir / "manifest.json"
    force_reexport = bool(dataset_cfg.get("force_reexport", False))

    current_gt = count_current_gt()
    if dataset_dir.exists() and manifest_file.exists() and not force_reexport:
        manifest = json.loads(manifest_file.read_text())
        exported_total = int(manifest.get("total", 0))
        print(f"Reusing dataset: {dataset_dir}")
        print(f"  Exported lines: {exported_total}")
        print(f"  Current GT lines in DB: {current_gt}")
        if current_gt > exported_total:
            print("  Note: DB has newer annotations than this frozen dataset. Change dataset_id or set force_reexport=true to include them.")
        return dataset_dir

    print(f"Exporting dataset: {dataset_id}")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "export_gt.py"),
        "--shared",
        str(shared),
        "--dataset-id",
        dataset_id,
    ]
    if force_reexport:
        cmd.append("--force")
    result = run_cmd(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise AutomationError(result.stderr or result.stdout)
    print(result.stdout.strip())
    return dataset_dir


def sync_dataset_to_gpu_cache(cfg: dict, dataset_dir: Path) -> None:
    dataset_cfg = cfg.get("dataset", {})
    if not bool(dataset_cfg.get("sync_to_gpu_cache", False)):
        print("GPU cache sync disabled")
        return

    gpu_cfg = cfg.get("gpu", {})
    remote_root = gpu_cfg.get("remote_dataset_cache_root")
    if not remote_root:
        print("GPU cache sync skipped: remote_dataset_cache_root is not configured")
        return

    ssh_target = get_ssh_target(cfg)
    port = str(gpu_cfg.get("port", 22))
    remote_dataset_dir = f"{remote_root.rstrip('/')}/{dataset_dir.name}/"

    mkdir_result = run_cmd(["ssh", "-p", port, ssh_target, "mkdir", "-p", remote_root])
    if mkdir_result.returncode != 0:
        raise AutomationError(mkdir_result.stderr or mkdir_result.stdout or "Failed to create remote dataset cache dir")

    print(f"Syncing dataset to GPU cache: {ssh_target}:{remote_dataset_dir}")
    rsync_result = run_cmd(
        [
            "rsync",
            "-az",
            "--delete",
            "-e",
            f"ssh -p {port}",
            f"{str(dataset_dir)}/",
            f"{ssh_target}:{remote_dataset_dir}",
        ]
    )
    if rsync_result.returncode != 0:
        raise AutomationError(rsync_result.stderr or rsync_result.stdout or "Dataset rsync failed")
    print("Dataset sync OK")


def submit_job(cfg: dict, shared: Path) -> str:
    job_cfg = cfg.get("job", {})
    job_type = job_cfg.get("type", "finetune")
    dataset_id = cfg.get("dataset", {}).get("dataset_id")
    now = datetime.now(timezone.utc)
    job_id = f"{job_type}_{now.strftime('%Y%m%d_%H%M%S')}"
    job_dir = shared / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    dataset_path = str(shared / "datasets" / dataset_id)
    resume_from = str(job_cfg.get("resume_from", "") or "").strip()
    eval_checkpoint = str(job_cfg.get("eval_checkpoint", "") or "").strip()

    if job_type == "eval" and not eval_checkpoint:
        raise AutomationError("automation.toml: [job].eval_checkpoint is required when job.type = 'eval'")

    job = {
        "job_id": job_id,
        "type": job_type,
        "label": job_cfg.get("label") or job_id,
        "created_at": now.isoformat(),
        "dataset_id": dataset_id,
        "dataset_path": dataset_path,
        "base_model": job_cfg.get("base_model", "microsoft/trocr-base-handwritten"),
        "resume_from": resume_from or None,
        "eval_checkpoint": eval_checkpoint or None,
        "params": {
            "epochs": int(job_cfg.get("epochs", 10)),
            "batch_size": int(job_cfg.get("batch_size", 8)),
            "learning_rate": float(job_cfg.get("learning_rate", 5e-5)),
        },
    }
    (job_dir / "job.json").write_text(json.dumps(job, indent=2))
    (job_dir / "progress.json").write_text(
        json.dumps(
            {
                "status": "pending",
                "message": "Waiting for GPU worker",
                "updated_at": now.isoformat(),
            },
            indent=2,
        )
    )
    return job_id


def main() -> None:
    try:
        cfg = load_config()
        shared = get_shared_path(cfg)

        print_step("Prechecks")
        precheck_shared_path(shared)
        if bool(cfg.get("prechecks", {}).get("require_ping", True)):
            ping_gpu(cfg)

        worker_status = load_worker_status(shared)
        if bool(cfg.get("prechecks", {}).get("require_worker_heartbeat", False)):
            if worker_status is None:
                raise AutomationError("GPU worker heartbeat file is missing. Start scripts/gpu_worker.py on the GPU machine first.")
            age = heartbeat_age_seconds(worker_status)
            max_age = int(cfg.get("prechecks", {}).get("worker_heartbeat_max_age_seconds", 120))
            if age is None or age > max_age:
                raise AutomationError(
                    f"GPU worker heartbeat is stale ({'unknown' if age is None else int(age)}s)."
                )
            print(f"Worker heartbeat OK ({int(age)}s old, status={worker_status.get('status')})")
        elif worker_status is not None:
            age = heartbeat_age_seconds(worker_status)
            if age is not None:
                print(f"Worker status: {worker_status.get('status', 'unknown')} ({int(age)}s old)")

        if bool(cfg.get("prechecks", {}).get("require_no_active_jobs", True)):
            jobs = active_jobs(shared)
            if jobs:
                details = "; ".join(f"{j['job_id']} [{j['status']}]" for j in jobs)
                raise AutomationError(f"Active jobs already exist: {details}")
            print("No active jobs found")

        print_step("Incremental split freeze")
        assign_splits()

        print_step("Dataset export / reuse")
        dataset_dir = ensure_dataset(cfg, shared)

        print_step("Optional GPU dataset sync")
        sync_dataset_to_gpu_cache(cfg, dataset_dir)

        print_step("Submit job")
        job_id = submit_job(cfg, shared)
        print(f"Job submitted: {job_id}")
        print(f"Monitor: /jobs  or  {shared / 'jobs' / job_id}")
    except AutomationError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
