#!/usr/bin/env python3
"""
Sync latest local code to the GPU server over SSH.

Behavior:
1. Read SSH + paths from automation.toml
2. Check for ongoing computation (shared job state + optional remote process check)
3. Rsync repo to remote destination

Usage:
    python scripts/sync_code_to_gpu.py
    python scripts/sync_code_to_gpu.py --force
    python scripts/sync_code_to_gpu.py --skip-remote-process-check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "automation.toml"
ACTIVE_STATUSES = {"pending", "running"}


class SyncError(RuntimeError):
    pass


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SyncError(f"Missing config file: {CONFIG_PATH}")
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def get_ssh_target(cfg: dict) -> tuple[str, str]:
    gpu = cfg.get("gpu", {})
    host = str(gpu.get("host") or "").strip()
    user = str(gpu.get("user") or "").strip()
    if not host:
        raise SyncError("automation.toml: [gpu].host is required")
    if not user:
        raise SyncError("automation.toml: [gpu].user is required for sync")
    return f"{user}@{host}", str(gpu.get("port", 22))


def get_shared_path(cfg: dict) -> Path:
    shared = str(cfg.get("shared", {}).get("path") or "").strip()
    if not shared:
        raise SyncError("automation.toml: [shared].path is required")
    p = Path(shared).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise SyncError(f"Shared path is not accessible: {p}")
    return p


def get_code_destination(cfg: dict) -> str:
    dest = str(cfg.get("gpu", {}).get("code_destination") or "").strip()
    if not dest:
        raise SyncError("automation.toml: [gpu].code_destination is required")
    return dest


def check_shared_active_jobs(shared: Path) -> list[str]:
    jobs_dir = shared / "jobs"
    active: list[str] = []
    if jobs_dir.exists():
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
            status = progress.get("status")
            if status in ACTIVE_STATUSES:
                active.append(f"{job.get('job_id', job_dir.name)} [{status}]")

    worker_status = shared / "worker_status.json"
    if worker_status.exists():
        try:
            ws = json.loads(worker_status.read_text())
            if ws.get("status") == "running":
                active.append(f"worker_status running (current_job={ws.get('current_job')})")
        except Exception:
            pass

    return active


def check_remote_processes(ssh_target: str, port: str) -> list[str]:
    cmd = [
        "ssh",
        "-p",
        port,
        ssh_target,
        (
            "pgrep -af \"(scripts/gpu_worker.py|scripts/finetune_trocr.py|scripts/eval_model.py|python.*finetune_trocr|python.*eval_model)\" "
            "| grep -v \"pgrep -af\" || true"
        ),
    ]
    res = run_cmd(cmd)
    if res.returncode != 0:
        raise SyncError(f"Remote process check failed: {res.stderr or res.stdout}")
    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    return lines


def ensure_remote_destination(ssh_target: str, port: str, remote_dest: str) -> None:
    cmd = ["ssh", "-p", port, ssh_target, "mkdir", "-p", remote_dest]
    res = run_cmd(cmd)
    if res.returncode != 0:
        raise SyncError(f"Could not create remote destination: {res.stderr or res.stdout}")


def sync_repo(ssh_target: str, port: str, remote_dest: str) -> None:
    excludes = [
        ".git/",
        ".venv/",
        "working/",
        "__pycache__/",
        "*.pyc",
        ".mypy_cache/",
        ".pytest_cache/",
    ]

    cmd = ["rsync", "-az", "--delete", "-e", f"ssh -p {port}"]
    for ex in excludes:
        cmd += ["--exclude", ex]

    cmd += [f"{str(PROJECT_ROOT)}/", f"{ssh_target}:{remote_dest.rstrip('/')}/"]
    res = run_cmd(cmd)
    if res.returncode != 0:
        raise SyncError(f"Code sync failed: {res.stderr or res.stdout}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local code to GPU server")
    parser.add_argument("--force", action="store_true", help="Sync even if active jobs/processes are detected")
    parser.add_argument(
        "--skip-remote-process-check",
        action="store_true",
        help="Skip ssh process check (shared-job check still runs)",
    )
    args = parser.parse_args()

    try:
        cfg = load_config()
        ssh_target, port = get_ssh_target(cfg)
        shared = get_shared_path(cfg)
        remote_dest = get_code_destination(cfg)

        print("==> Prechecks")
        active_shared = check_shared_active_jobs(shared)
        if active_shared:
            print("Active work from shared state:")
            for item in active_shared:
                print(f"  - {item}")
        else:
            print("No active jobs found in shared state")

        active_remote: list[str] = []
        if not args.skip_remote_process_check:
            active_remote = check_remote_processes(ssh_target, port)
            if active_remote:
                print("Active processes on remote GPU server:")
                for item in active_remote:
                    print(f"  - {item}")
            else:
                print("No matching training/worker processes found on remote")

        if (active_shared or active_remote) and not args.force:
            raise SyncError("Ongoing computation detected. Re-run with --force to sync anyway.")

        print("\n==> Sync code")
        ensure_remote_destination(ssh_target, port, remote_dest)
        sync_repo(ssh_target, port, remote_dest)

        print("Sync complete")
        print(f"  local:  {PROJECT_ROOT}")
        print(f"  remote: {ssh_target}:{remote_dest}")
    except SyncError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
