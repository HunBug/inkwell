from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path

from flask import Blueprint, render_template, current_app

jobs_bp = Blueprint("jobs", __name__, url_prefix="/jobs")

SHARED_ENV = "INKWELL_SHARED"


def _get_shared_path() -> Path:
    override = current_app.config.get("INKWELL_SHARED") or os.environ.get(SHARED_ENV)
    if override:
        return Path(override).expanduser().resolve()

    # Fallback 1: automation.toml shared.path (keeps UI in sync with launcher config)
    try:
        from inkwell.db import PROJECT_ROOT

        cfg_path = PROJECT_ROOT / "automation.toml"
        if cfg_path.exists():
            with open(cfg_path, "rb") as f:
                cfg = tomllib.load(f)
            configured = cfg.get("shared", {}).get("path")
            if configured:
                return Path(configured).expanduser().resolve()
    except Exception:
        pass

    # Fallback 2: local dev path
    from inkwell.db import DEFAULT_DB_PATH
    return DEFAULT_DB_PATH.parent / "shared"


def _load_jobs() -> list[dict]:
    shared = _get_shared_path()
    jobs_dir = shared / "jobs"
    if not jobs_dir.exists():
        return []

    jobs = []
    for d in sorted(jobs_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        job_file = d / "job.json"
        progress_file = d / "progress.json"
        if not job_file.exists():
            continue

        try:
            job = json.loads(job_file.read_text())
        except Exception:
            continue

        progress: dict = {}
        if progress_file.exists():
            try:
                progress = json.loads(progress_file.read_text())
            except Exception:
                pass

        result: dict = {}
        result_file = d / "result.json"
        if result_file.exists():
            try:
                result = json.loads(result_file.read_text())
            except Exception:
                pass

        log_tail: list[str] = []
        log_file = d / "worker.log"
        if log_file.exists():
            try:
                lines = log_file.read_text(errors="replace").splitlines()
                log_tail = lines[-20:]
            except Exception:
                pass

        can_cancel = progress.get("status") in ("pending", "running")
        cancel_exists = (d / "CANCEL").exists()

        jobs.append({
            "job_id": job.get("job_id", d.name),
            "label": job.get("label", d.name),
            "type": job.get("type", "?"),
            "dataset_id": job.get("dataset_id", "?"),
            "base_model": job.get("base_model", ""),
            "created_at": job.get("created_at", ""),
            "params": job.get("params", {}),
            "status": progress.get("status", "unknown"),
            "progress": progress,
            "result": result,
            "log_tail": log_tail,
            "can_cancel": can_cancel and not cancel_exists,
            "cancel_requested": cancel_exists,
            "job_dir": str(d),
        })

    return jobs


def _load_worker_status() -> dict | None:
    shared = _get_shared_path()
    status_file = shared / "worker_status.json"
    if not status_file.exists():
        return None
    try:
        return json.loads(status_file.read_text())
    except Exception:
        return None


def _is_pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        proc_cmdline = Path(f"/proc/{pid}/cmdline")
        if proc_cmdline.exists():
            cmd = proc_cmdline.read_text(errors="ignore").replace("\x00", " ")
            return "run_automation.py" in cmd
        return True
    except OSError:
        return False


def _launcher_paths(shared: Path) -> tuple[Path, Path]:
    control_dir = shared / "control"
    return control_dir / "automation_launcher_state.json", control_dir / "automation_launcher.log"


def _detect_launcher_status_from_log(log_path: Path) -> str:
    if not log_path.exists():
        return "unknown"
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        tail = lines[-50:]
        if any(line.startswith("ERROR:") for line in tail):
            return "failed"
        if any("Job submitted:" in line for line in tail):
            return "completed"
    except Exception:
        return "unknown"
    return "unknown"


def _load_launcher_state(shared: Path) -> dict:
    state_path, log_path = _launcher_paths(shared)
    if not state_path.parent.exists():
        return {
            "status": "idle",
            "pid": None,
            "started_at": None,
            "updated_at": None,
            "last_result": None,
            "log_tail": [],
        }

    if not state_path.exists():
        return {
            "status": "idle",
            "pid": None,
            "started_at": None,
            "updated_at": None,
            "last_result": None,
            "log_tail": [],
        }

    try:
        state = json.loads(state_path.read_text())
    except Exception:
        state = {
            "status": "idle",
            "pid": None,
            "started_at": None,
            "updated_at": None,
            "last_result": None,
        }

    if state.get("status") == "running" and not _is_pid_running(state.get("pid")):
        final_status = _detect_launcher_status_from_log(log_path)
        state["status"] = final_status
        state["pid"] = None
        state["last_result"] = final_status
        state["updated_at"] = datetime.utcnow().isoformat() + "+00:00"
        state_path.write_text(json.dumps(state, indent=2))

    log_tail: list[str] = []
    if log_path.exists():
        try:
            log_tail = log_path.read_text(errors="replace").splitlines()[-30:]
        except Exception:
            pass

    state["log_tail"] = log_tail
    return state


def _load_automation_config_summary() -> dict:
    from inkwell.db import PROJECT_ROOT

    cfg_path = PROJECT_ROOT / "automation.toml"
    summary = {
        "config_path": str(cfg_path),
        "exists": cfg_path.exists(),
        "shared_path": None,
        "gpu_host": None,
        "dataset_id": None,
        "job_type": None,
        "sync_to_gpu_cache": None,
    }
    if not cfg_path.exists():
        return summary
    try:
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)
        summary.update(
            {
                "shared_path": cfg.get("shared", {}).get("path"),
                "gpu_host": cfg.get("gpu", {}).get("host"),
                "dataset_id": cfg.get("dataset", {}).get("dataset_id"),
                "job_type": cfg.get("job", {}).get("type"),
                "sync_to_gpu_cache": cfg.get("dataset", {}).get("sync_to_gpu_cache"),
            }
        )
    except Exception:
        pass
    return summary


def _start_automation_launcher(shared: Path) -> bool:
    from inkwell.db import PROJECT_ROOT

    state_path, log_path = _launcher_paths(shared)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_launcher_state(shared)
    if current.get("status") == "running":
        return False

    with open(log_path, "a", encoding="utf-8") as log_fh:
        log_fh.write("\n" + "=" * 80 + "\n")
        log_fh.write(f"Start: {datetime.utcnow().isoformat()}+00:00\n")
        log_fh.flush()

        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "run_automation.py")],
            cwd=str(PROJECT_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

    state = {
        "status": "running",
        "pid": proc.pid,
        "started_at": datetime.utcnow().isoformat() + "+00:00",
        "updated_at": datetime.utcnow().isoformat() + "+00:00",
        "last_result": None,
    }
    state_path.write_text(json.dumps(state, indent=2))
    return True


def _age(iso_str: str) -> str:
    """Human-readable age from an ISO timestamp."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        delta = datetime.now(dt.tzinfo) - dt
        s = int(delta.total_seconds())
        if s < 60:
            return f"{s}s ago"
        if s < 3600:
            return f"{s//60}m ago"
        if s < 86400:
            return f"{s//3600}h ago"
        return f"{s//86400}d ago"
    except Exception:
        return ""


@jobs_bp.route("/")
def index():
    shared = _get_shared_path()
    jobs = _load_jobs()
    jobs_active = any(j["status"] in ("pending", "running") for j in jobs)
    shared_path = str(shared)
    worker_status = _load_worker_status()
    launcher = _load_launcher_state(shared)
    any_running = jobs_active or launcher.get("status") == "running"
    cfg_summary = _load_automation_config_summary()
    pending_count = sum(1 for j in jobs if j["status"] == "pending")
    running_count = sum(1 for j in jobs if j["status"] == "running")
    return render_template(
        "jobs.html",
        jobs=jobs,
        any_running=any_running,
        shared_path=shared_path,
        worker_status=worker_status,
        launcher=launcher,
        cfg_summary=cfg_summary,
        pending_count=pending_count,
        running_count=running_count,
        age=_age,
    )


@jobs_bp.route("/<job_id>/cancel", methods=["POST"])
def cancel(job_id: str):
    from flask import redirect, url_for
    shared = _get_shared_path()
    job_dir = shared / "jobs" / job_id
    cancel_file = job_dir / "CANCEL"
    if job_dir.exists() and not cancel_file.exists():
        cancel_file.touch()
    return redirect(url_for("jobs.index"))


@jobs_bp.route("/automation/run", methods=["POST"])
def run_automation():
    from flask import redirect, url_for

    shared = _get_shared_path()
    _start_automation_launcher(shared)
    return redirect(url_for("jobs.index"))
