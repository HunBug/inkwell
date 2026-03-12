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
import shlex
import subprocess
import sys
from pathlib import Path

import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "automation.toml"
ACTIVE_STATUSES = {"running"}


class SyncError(RuntimeError):
    pass


def run_cmd(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SyncError(f"Command timed out after {timeout}s: {' '.join(cmd)}") from exc


def ssh_base_cmd(port: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-p",
        port,
    ]


def ssh_bash_lc_cmd(port: str, ssh_target: str, script: str) -> list[str]:
    return ssh_base_cmd(port) + [ssh_target, f"bash -lc {shlex.quote(script)}"]


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


def get_remote_shared_path(cfg: dict) -> str:
    shared = str(cfg.get("gpu", {}).get("shared_path_on_server") or "").strip()
    if not shared:
        raise SyncError("automation.toml: [gpu].shared_path_on_server is required to start the remote worker")
    return shared


def get_remote_worker_local_datasets(cfg: dict) -> str | None:
    value = str(
        cfg.get("gpu", {}).get("worker_local_datasets")
        or cfg.get("gpu", {}).get("remote_dataset_cache_root")
        or ""
    ).strip()
    return value or None


def expand_remote_path(ssh_target: str, port: str, raw_path: str) -> str:
    remote_cmd = (
        "python3 -c "
        + shlex.quote("from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser())")
        + " "
        + shlex.quote(raw_path)
    )
    cmd = ssh_base_cmd(port) + [ssh_target, remote_cmd]
    res = run_cmd(cmd, timeout=20)
    if res.returncode != 0:
        raise SyncError(f"Failed to expand remote path {raw_path}: {res.stderr or res.stdout}")
    return res.stdout.strip()


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
    cmd = ssh_base_cmd(port) + [
        ssh_target,
        (
            "pgrep -af \"(scripts/gpu_worker.py|scripts/finetune_trocr.py|scripts/eval_model.py|python.*finetune_trocr|python.*eval_model)\" "
            "| grep -v \"pgrep -af\" || true"
        ),
    ]
    res = run_cmd(cmd, timeout=20)
    if res.returncode != 0:
        raise SyncError(f"Remote process check failed: {res.stderr or res.stdout}")
    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    return lines


def check_remote_worker_processes(ssh_target: str, port: str) -> list[str]:
    cmd = ssh_base_cmd(port) + [
        ssh_target,
        "pgrep -af \"scripts/gpu_worker.py\" | grep -v \"pgrep -af\" || true",
    ]
    res = run_cmd(cmd, timeout=20)
    if res.returncode != 0:
        raise SyncError(f"Remote worker check failed: {res.stderr or res.stdout}")
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def stop_remote_worker(ssh_target: str, port: str, remote_dest: str) -> None:
    remote_dest_abs = expand_remote_path(ssh_target, port, remote_dest)
    pid_file = f"{remote_dest_abs.rstrip('/')}/working/gpu_worker.pid"
    remote_cmd = (
        "set +e; "
        f"PID_FILE={shlex.quote(pid_file)}; "
        "if [ -f \"$PID_FILE\" ]; then "
        "  pid=$(cat \"$PID_FILE\"); "
        "  kill \"$pid\" 2>/dev/null || true; "
        "  sleep 1; "
        "  kill -0 \"$pid\" 2>/dev/null && kill -9 \"$pid\" 2>/dev/null || true; "
        "  rm -f \"$PID_FILE\"; "
        "fi; "
        "pkill -f \"scripts/gpu_worker.py\" 2>/dev/null || true; "
        "exit 0"
    )
    cmd = ssh_bash_lc_cmd(port, ssh_target, remote_cmd)
    res = run_cmd(cmd, timeout=30)
    if res.returncode != 0:
        remaining = check_remote_worker_processes(ssh_target, port)
        if not remaining:
            return
        detail = (res.stderr or res.stdout or f"ssh exit code {res.returncode}").strip()
        raise SyncError(f"Failed to stop remote worker: {detail}")

    remaining = check_remote_worker_processes(ssh_target, port)
    if remaining:
        raise SyncError("Failed to stop remote worker: process still running after stop command")


def start_remote_worker(
    ssh_target: str,
    port: str,
    remote_dest: str,
    remote_shared_path: str,
    worker_local_datasets: str | None,
) -> str:
    remote_dest_abs = expand_remote_path(ssh_target, port, remote_dest)
    remote_shared_abs = expand_remote_path(ssh_target, port, remote_shared_path)
    worker_local_abs = expand_remote_path(ssh_target, port, worker_local_datasets) if worker_local_datasets else None

    remote_dest_q = shlex.quote(remote_dest_abs)
    remote_shared_q = shlex.quote(remote_shared_abs)
    python_bin_q = shlex.quote(f"{remote_dest_abs.rstrip('/')}/.venv/bin/python")
    log_path_q = shlex.quote(f"{remote_dest_abs.rstrip('/')}/working/gpu_worker.log")
    pid_file_q = shlex.quote(f"{remote_dest_abs.rstrip('/')}/working/gpu_worker.pid")
    local_datasets_q = shlex.quote(worker_local_abs) if worker_local_abs else None

    cmd_parts = [
        "set -e",
        f"mkdir -p {remote_dest_q}/working",
    ]
    if local_datasets_q:
        cmd_parts.append(f"mkdir -p {local_datasets_q}")

    worker_cmd = (
        f"nohup env INKWELL_SHARED={remote_shared_q} {python_bin_q} scripts/gpu_worker.py"
        + (f" --local-datasets {local_datasets_q}" if local_datasets_q else "")
        + f" >> {log_path_q} 2>&1 < /dev/null"
    )

    cmd_parts += [
        f"cd {remote_dest_q}",
        f"{worker_cmd} & pid=$!",
        f"echo \"$pid\" > {pid_file_q}",
        "sleep 1",
        f"if kill -0 \"$pid\" 2>/dev/null; then echo \"$pid\"; else tail -n 40 {log_path_q}; exit 1; fi",
    ]

    remote_cmd = "; ".join(cmd_parts)
    cmd = ssh_bash_lc_cmd(port, ssh_target, remote_cmd)
    res = run_cmd(cmd, timeout=30)
    if res.returncode != 0:
        raise SyncError(f"Failed to start remote worker: {res.stderr or res.stdout}")
    return (res.stdout or "").strip()


def prompt_yes_no_stop_or_abort() -> str:
    while True:
        answer = input("Remote worker is already running. [s]top and continue, or [a]bort? ").strip().lower()
        if answer in {"s", "stop"}:
            return "stop"
        if answer in {"a", "abort"}:
            return "abort"


def ensure_remote_destination(ssh_target: str, port: str, remote_dest: str) -> None:
    cmd = ssh_base_cmd(port) + [ssh_target, "mkdir", "-p", remote_dest]
    res = run_cmd(cmd, timeout=20)
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

    cmd = [
        "rsync",
        "-az",
        "--delete",
        "-e",
        f"ssh -o BatchMode=yes -o ConnectTimeout=8 -p {port}",
    ]
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
        "--start-runner",
        action="store_true",
        help="Start the remote gpu_worker.py after sync; if already running, prompt to stop or abort",
    )
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
        remote_shared_path = get_remote_shared_path(cfg) if args.start_runner else None
        worker_local_datasets = get_remote_worker_local_datasets(cfg) if args.start_runner else None

        print("==> Prechecks")
        active_shared = check_shared_active_jobs(shared)
        if active_shared:
            print("Active work from shared state:")
            for item in active_shared:
                print(f"  - {item}")
        else:
            print("No active jobs found in shared state")

        active_remote: list[str] = []
        remote_worker: list[str] = []
        if not args.skip_remote_process_check:
            active_remote = check_remote_processes(ssh_target, port)
            if active_remote:
                print("Active processes on remote GPU server:")
                for item in active_remote:
                    print(f"  - {item}")
            else:
                print("No matching training/worker processes found on remote")

            remote_worker = check_remote_worker_processes(ssh_target, port)
            if remote_worker:
                print("Remote worker process found:")
                for item in remote_worker:
                    print(f"  - {item}")

        restart_runner_after_sync = False
        if args.start_runner and remote_worker:
            if args.force:
                choice = "stop"
            else:
                choice = prompt_yes_no_stop_or_abort()
            if choice == "abort":
                raise SyncError("Aborted by user because remote worker is already running.")
            stop_remote_worker(ssh_target, port, remote_dest)
            restart_runner_after_sync = True
            print("Remote worker stopped; it will be restarted after sync")
        elif args.start_runner:
            restart_runner_after_sync = True

        active_non_worker_remote = [line for line in active_remote if "scripts/gpu_worker.py" not in line]
        if (active_shared or active_non_worker_remote) and not args.force:
            raise SyncError("Ongoing computation detected. Re-run with --force to sync anyway.")

        print("\n==> Sync code")
        remote_dest = expand_remote_path(ssh_target, port, remote_dest)
        ensure_remote_destination(ssh_target, port, remote_dest)
        sync_repo(ssh_target, port, remote_dest)

        if restart_runner_after_sync:
            print("\n==> Start remote worker")
            pid = start_remote_worker(
                ssh_target,
                port,
                remote_dest,
                remote_shared_path,
                worker_local_datasets,
            )
            print(f"Remote worker started (pid={pid or 'unknown'})")

        print("Sync complete")
        print(f"  local:  {PROJECT_ROOT}")
        print(f"  remote: {ssh_target}:{remote_dest}")
    except SyncError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
