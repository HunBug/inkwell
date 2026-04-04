from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
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

        telemetry_tail: list[str] = []
        telemetry_file = d / "telemetry.log"
        if telemetry_file.exists():
            try:
                lines = telemetry_file.read_text(errors="replace").splitlines()
                telemetry_tail = lines[-30:]
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
            "eval_checkpoint": job.get("eval_checkpoint", ""),
            "split": job.get("split", "val"),
            "resume_from": job.get("resume_from", ""),
            "created_at": job.get("created_at", ""),
            "params": job.get("params", {}),
            "status": progress.get("status", "unknown"),
            "progress": progress,
            "result": result,
            "log_tail": log_tail,
            "telemetry_tail": telemetry_tail,
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
        return True
    except OSError:
        return False


def _launcher_paths(shared: Path) -> tuple[Path, Path]:
    control_dir = shared / "control"
    return control_dir / "automation_launcher_state.json", control_dir / "automation_launcher.log"


def _sync_paths(shared: Path) -> tuple[Path, Path]:
    control_dir = shared / "control"
    return control_dir / "sync_launcher_state.json", control_dir / "sync_launcher.log"


def _pool_paths(shared: Path) -> tuple[Path, Path]:
    control_dir = shared / "control"
    return control_dir / "pool_launcher_state.json", control_dir / "pool_launcher.log"


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


def _detect_sync_status_from_log(log_path: Path) -> str:
    if not log_path.exists():
        return "unknown"
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        tail = lines[-80:]
        if any(line.startswith("ERROR:") for line in tail):
            return "failed"
        if any("Sync complete" in line for line in tail):
            return "completed"
    except Exception:
        return "unknown"
    return "unknown"


def _detect_pool_status_from_log(log_path: Path) -> str:
    if not log_path.exists():
        return "unknown"
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        tail = lines[-120:]
        if any(line.startswith("ERROR:") for line in tail):
            return "failed"
        if any("Done: export completed and infer_pool job submitted." in line for line in tail):
            return "completed"
        if any("Job submitted: infer_pool_" in line for line in tail):
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


def _load_sync_state(shared: Path) -> dict:
    state_path, log_path = _sync_paths(shared)
    if not state_path.parent.exists() or not state_path.exists():
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
        final_status = _detect_sync_status_from_log(log_path)
        state["status"] = final_status
        state["pid"] = None
        state["last_result"] = final_status
        state["updated_at"] = datetime.utcnow().isoformat() + "+00:00"
        state_path.write_text(json.dumps(state, indent=2))

    log_tail: list[str] = []
    if log_path.exists():
        try:
            log_tail = log_path.read_text(errors="replace").splitlines()[-60:]
        except Exception:
            pass

    state["log_tail"] = log_tail
    return state


def _load_pool_state(shared: Path) -> dict:
    state_path, log_path = _pool_paths(shared)
    if not state_path.parent.exists() or not state_path.exists():
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
        final_status = _detect_pool_status_from_log(log_path)
        state["status"] = final_status
        state["pid"] = None
        state["last_result"] = final_status
        state["updated_at"] = datetime.utcnow().isoformat() + "+00:00"
        state_path.write_text(json.dumps(state, indent=2))

    log_tail: list[str] = []
    if log_path.exists():
        try:
            log_tail = log_path.read_text(errors="replace").splitlines()[-80:]
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


def _start_sync_launcher(shared: Path) -> bool:
    from inkwell.db import PROJECT_ROOT

    state_path, log_path = _sync_paths(shared)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_sync_state(shared)
    if current.get("status") == "running":
        return False

    with open(log_path, "a", encoding="utf-8") as log_fh:
        log_fh.write("\n" + "=" * 80 + "\n")
        log_fh.write(f"Start: {datetime.utcnow().isoformat()}+00:00\n")
        log_fh.flush()

        proc = subprocess.Popen(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "sync_code_to_gpu.py"),
                "--force",
                "--start-runner",
            ],
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


def _start_pool_launcher(shared: Path) -> bool:
    from inkwell.db import PROJECT_ROOT

    state_path, log_path = _pool_paths(shared)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_pool_state(shared)
    if current.get("status") == "running":
        return False

    with open(log_path, "a", encoding="utf-8") as log_fh:
        log_fh.write("\n" + "=" * 80 + "\n")
        log_fh.write(f"Start: {datetime.utcnow().isoformat()}+00:00\n")
        log_fh.flush()

        proc = subprocess.Popen(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_pool_infer.py"),
            ],
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


def _run_pick_suggestions(shared: Path, n: int = 150) -> tuple[bool, str]:
    from inkwell.db import PROJECT_ROOT

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "pick_next_samples.py"),
        "--n",
        str(n),
        "--shared",
        str(shared),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        return False, f"Failed to start picker: {e}"

    output_parts = (proc.stdout or "", proc.stderr or "")
    combined_out = "\n".join(
        part.strip()
        for part in output_parts
        if part and part.strip()
    )

    if proc.returncode != 0:
        if combined_out:
            tail = "\n".join(combined_out.splitlines()[-4:])
        else:
            tail = "unknown error"
        return False, f"Suggestion picker failed ({proc.returncode}): {tail}"

    written_path = None
    for line in combined_out.splitlines():
        marker = "[pick] Written:"
        if marker in line:
            written_path = line.split(marker, 1)[1].strip()
            break

    if written_path:
        return True, f"Suggestions generated: {Path(written_path).name}"
    return True, "Suggestions generated successfully."


BASELINE_MODEL = "microsoft/trocr-base-handwritten"


def _load_datasets_with_eval_status(shared: Path, jobs: list[dict]) -> list[dict]:
    """Scan shared/datasets/ and annotate each with baseline + finetuned eval status."""
    datasets_dir = shared / "datasets"
    if not datasets_dir.exists():
        return []

    # Index eval jobs and completed finetune jobs by dataset_id
    eval_by_dataset: dict[str, list[dict]] = {}
    finetuned_by_dataset: dict[str, list[dict]] = {}
    for job in jobs:
        ds_id = job.get("dataset_id", "")
        if job.get("type") == "eval":
            eval_by_dataset.setdefault(ds_id, []).append(job)
        elif job.get("type") == "finetune" and job.get("status") == "completed":
            finetuned_by_dataset.setdefault(ds_id, []).append(job)

    result = []
    for d in sorted(datasets_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not d.is_dir():
            continue
        manifest: dict = {}
        manifest_file = d / "manifest.json"
        if manifest_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text())
            except Exception:
                pass

        dataset_id = d.name
        eval_jobs = eval_by_dataset.get(dataset_id, [])

        # --- Baseline eval status by split ---
        baseline_eval: dict[str, dict | None] = {"val": None, "test": None}
        baseline_eval_pending: dict[str, bool] = {"val": False, "test": False}
        for split in ("val", "test"):
            completed = [
                ej for ej in eval_jobs
                if (ej.get("split", "val") == split)
                and (BASELINE_MODEL in (ej.get("eval_checkpoint", "") or "") or (ej.get("eval_checkpoint") or "") == BASELINE_MODEL)
                and ej.get("status") == "completed"
            ]
            if completed:
                completed = sorted(completed, key=lambda x: x.get("created_at", ""), reverse=True)
                baseline_eval[split] = completed[0]

            baseline_eval_pending[split] = any(
                (ej.get("split", "val") == split)
                and (BASELINE_MODEL in (ej.get("eval_checkpoint", "") or "") or (ej.get("eval_checkpoint") or "") == BASELINE_MODEL)
                and ej.get("status") in ("pending", "running")
                for ej in eval_jobs
            )

        # --- Fine-tuned checkpoints with their eval status ---
        finetuned_with_eval = []
        for ft_job in finetuned_by_dataset.get(dataset_id, []):
            ft_id = ft_job.get("job_id", "")
            cp_suffix = f"{ft_id}/checkpoints/best"
            ft_eval: dict[str, dict | None] = {"val": None, "test": None}
            ft_eval_pending: dict[str, bool] = {"val": False, "test": False}
            for split in ("val", "test"):
                completed = [
                    ej for ej in eval_jobs
                    if (ej.get("split", "val") == split)
                    and cp_suffix in (ej.get("eval_checkpoint", "") or "")
                    and ej.get("status") == "completed"
                ]
                if completed:
                    completed = sorted(completed, key=lambda x: x.get("created_at", ""), reverse=True)
                    ft_eval[split] = completed[0]

                ft_eval_pending[split] = any(
                    (ej.get("split", "val") == split)
                    and cp_suffix in (ej.get("eval_checkpoint", "") or "")
                    and ej.get("status") in ("pending", "running")
                    for ej in eval_jobs
                )
            finetuned_with_eval.append({
                "job_id": ft_id,
                "created_at": ft_job.get("created_at", ""),
                "eval": ft_eval,
                "eval_pending": ft_eval_pending,
            })

        counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}

        result.append({
            "dataset_id": dataset_id,
            "manifest": manifest,
            "train_count": counts.get("train", "?"),
            "val_count": counts.get("val", "?"),
            "test_count": counts.get("test", "?"),
            "baseline_eval": baseline_eval,
            "baseline_eval_pending": baseline_eval_pending,
            "finetuned": finetuned_with_eval,
        })

    return result


def _submit_eval_job(shared: Path, dataset_id: str, checkpoint: str, split: str = "val") -> str:
    """Write an eval job into the shared jobs folder.  Returns the new job_id."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    job_id = f"eval_{ts}"
    job_dir = shared / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Human-readable label: use last meaningful path component
    cp_label = Path(checkpoint).parent.parent.name if "/checkpoints/" in checkpoint else checkpoint.split("/")[-1]
    label = f"Eval {dataset_id} / {cp_label}"

    job = {
        "job_id": job_id,
        "type": "eval",
        "label": label,
        "dataset_id": dataset_id,
        "eval_checkpoint": checkpoint,
        "split": split,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (job_dir / "job.json").write_text(json.dumps(job, indent=2))
    (job_dir / "progress.json").write_text(json.dumps({
        "status": "pending",
        "message": "Waiting for GPU worker",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    return job_id


def _build_results_summary(shared: Path, jobs: list[dict]) -> list[dict]:
    datasets_dir = shared / "datasets"
    if not datasets_dir.exists():
        return []

    eval_jobs = [j for j in jobs if j.get("type") == "eval"]
    finetune_jobs = [
        j for j in jobs
        if j.get("type") == "finetune" and j.get("status") == "completed"
    ]

    summaries: list[dict] = []
    for d in sorted(datasets_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not d.is_dir():
            continue

        manifest: dict = {}
        mf = d / "manifest.json"
        if mf.exists():
            try:
                manifest = json.loads(mf.read_text())
            except Exception:
                manifest = {}

        dataset_id = d.name
        counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
        policy = manifest.get("text_policy") if isinstance(manifest.get("text_policy"), dict) else {}

        dataset_finetunes = sorted(
            [j for j in finetune_jobs if j.get("dataset_id") == dataset_id],
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )
        dataset_evals = [j for j in eval_jobs if j.get("dataset_id") == dataset_id]

        baseline_evals: dict[str, dict] = {}
        for split in ("val", "test"):
            completed = [
                e for e in dataset_evals
                if (e.get("split", "val") == split)
                and e.get("status") == "completed"
                and (e.get("eval_checkpoint") == BASELINE_MODEL or BASELINE_MODEL in (e.get("eval_checkpoint") or ""))
            ]
            if completed:
                completed = sorted(completed, key=lambda x: x.get("created_at", ""), reverse=True)
                baseline_evals[split] = completed[0]

        finetune_rows: list[dict] = []
        for ft in dataset_finetunes:
            ft_id = ft.get("job_id", "")
            cp_suffix = f"{ft_id}/checkpoints/best"
            eval_by_split: dict[str, dict] = {}
            for split in ("val", "test"):
                completed = [
                    e for e in dataset_evals
                    if (e.get("split", "val") == split)
                    and e.get("status") == "completed"
                    and cp_suffix in (e.get("eval_checkpoint") or "")
                ]
                if completed:
                    completed = sorted(completed, key=lambda x: x.get("created_at", ""), reverse=True)
                    eval_by_split[split] = completed[0]

            finetune_rows.append({
                "job_id": ft_id,
                "label": ft.get("label", ""),
                "base_model": ft.get("base_model", ""),
                "epochs": (ft.get("params") or {}).get("epochs") or (ft.get("result") or {}).get("epochs"),
                "created_at": ft.get("created_at"),
                "train_val_cer": (ft.get("result") or {}).get("final_val_cer"),
                "train_loss": (ft.get("result") or {}).get("final_train_loss"),
                "eval": eval_by_split,
            })

        best_val_eval = None
        best_val_cer = None
        for row in finetune_rows:
            ev = row.get("eval", {}).get("val")
            if not ev:
                continue
            cer = (ev.get("result") or {}).get("cer")
            if cer is None:
                continue
            if best_val_cer is None or cer < best_val_cer:
                best_val_cer = cer
                best_val_eval = row

        summaries.append({
            "dataset_id": dataset_id,
            "manifest": manifest,
            "counts": {
                "train": counts.get("train", "?"),
                "val": counts.get("val", "?"),
                "test": counts.get("test", "?"),
                "total": manifest.get("total", "?"),
            },
            "policy_name": policy.get("name"),
            "policy_dropped": manifest.get("dropped_by_text_policy"),
            "policy_transformed": manifest.get("transformed_by_text_policy"),
            "baseline_evals": baseline_evals,
            "finetunes": finetune_rows,
            "best_val_eval": best_val_eval,
            "latest_finetune": finetune_rows[0] if finetune_rows else None,
        })

    return summaries


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
    from flask import request

    shared = _get_shared_path()
    jobs = _load_jobs()
    jobs_active = any(j["status"] in ("pending", "running") for j in jobs)
    shared_path = str(shared)
    worker_status = _load_worker_status()
    launcher = _load_launcher_state(shared)
    sync = _load_sync_state(shared)
    pool = _load_pool_state(shared)
    any_running = (
        jobs_active
        or launcher.get("status") == "running"
        or sync.get("status") == "running"
        or pool.get("status") == "running"
    )
    cfg_summary = _load_automation_config_summary()
    pending_count = sum(1 for j in jobs if j["status"] == "pending")
    running_count = sum(1 for j in jobs if j["status"] == "running")
    datasets = _load_datasets_with_eval_status(shared, jobs)
    control_msg = request.args.get("control_msg")
    control_level = request.args.get("control_level", "ok")
    return render_template(
        "jobs.html",
        jobs=jobs,
        datasets=datasets,
        any_running=any_running,
        shared_path=shared_path,
        worker_status=worker_status,
        launcher=launcher,
        sync=sync,
        pool=pool,
        cfg_summary=cfg_summary,
        pending_count=pending_count,
        running_count=running_count,
        baseline_model=BASELINE_MODEL,
        control_msg=control_msg,
        control_level=control_level,
        age=_age,
    )


@jobs_bp.route("/results")
def results():
    shared = _get_shared_path()
    jobs = _load_jobs()
    summaries = _build_results_summary(shared, jobs)
    return render_template(
        "jobs_results.html",
        shared_path=str(shared),
        summaries=summaries,
        baseline_model=BASELINE_MODEL,
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


@jobs_bp.route("/sync/run", methods=["POST"])
def run_sync():
    from flask import redirect, url_for

    shared = _get_shared_path()
    _start_sync_launcher(shared)
    return redirect(url_for("jobs.index"))


@jobs_bp.route("/pool/run", methods=["POST"])
def run_pool():
    from flask import redirect, url_for

    shared = _get_shared_path()
    _start_pool_launcher(shared)
    return redirect(url_for("jobs.index"))


@jobs_bp.route("/suggestions/run", methods=["POST"])
def run_suggestions():
    from flask import request, redirect, url_for

    shared = _get_shared_path()
    raw_n = request.form.get("n", "150")
    try:
        n = int(raw_n)
    except Exception:
        n = 150
    n = max(1, min(n, 1000))

    ok, msg = _run_pick_suggestions(shared, n=n)
    level = "ok" if ok else "error"
    return redirect(
        url_for("jobs.index", control_msg=msg, control_level=level)
    )


@jobs_bp.route("/image/<dataset_id>/<path:image_path>")
def serve_dataset_image(dataset_id: str, image_path: str):
    """Serve a crop image from shared/datasets/<dataset_id>/<image_path>."""
    from flask import abort
    import re

    # Guard: dataset_id must be a plain slug, image_path must not escape the dir
    if not re.match(r'^[\w\-]+$', dataset_id):
        abort(400)
    shared = _get_shared_path()
    crops_root = (shared / "datasets" / dataset_id).resolve()
    target = (crops_root / image_path).resolve()
    # Prevent path traversal
    try:
        target.relative_to(crops_root)
    except ValueError:
        abort(400)
    if not target.exists():
        abort(404)
    from flask import send_file
    return send_file(target)


@jobs_bp.route("/<job_id>/eval-detail")
def eval_detail(job_id: str):
    from flask import abort
    import re

    if not re.match(r'^[\w\-]+$', job_id):
        abort(400)

    shared = _get_shared_path()
    job_dir = shared / "jobs" / job_id

    if not job_dir.exists():
        abort(404)

    try:
        job = json.loads((job_dir / "job.json").read_text())
    except Exception:
        abort(404)

    result: dict = {}
    result_file = job_dir / "result.json"
    if result_file.exists():
        try:
            result = json.loads(result_file.read_text())
        except Exception:
            pass

    per_line: list[dict] = sorted(
        result.get("per_line") or [],
        key=lambda x: x.get("cer", 0),
        reverse=True,
    )

    dataset_id = job.get("dataset_id", "")

    # Try to resolve the finetune job that produced this checkpoint, to get
    # base_model and label. Checkpoint path looks like: .../jobs/<ft_id>/checkpoints/best
    finetune_job: dict = {}
    checkpoint = job.get("eval_checkpoint") or result.get("checkpoint", "")
    if checkpoint and "checkpoints" in checkpoint:
        parts = Path(checkpoint).parts
        try:
            cp_idx = parts.index("checkpoints")
            ft_job_id = parts[cp_idx - 1]
            ft_job_file = shared / "jobs" / ft_job_id / "job.json"
            if ft_job_file.exists():
                finetune_job = json.loads(ft_job_file.read_text())
        except Exception:
            pass

    return render_template(
        "eval_detail.html",
        job=job,
        result=result,
        per_line=per_line,
        dataset_id=dataset_id,
        finetune_job=finetune_job,
        age=_age,
    )


@jobs_bp.route("/eval/submit", methods=["POST"])
def submit_eval():
    from flask import request, redirect, url_for

    shared = _get_shared_path()
    dataset_id = request.form["dataset_id"]
    checkpoint = request.form["checkpoint"]
    split = request.form.get("split", "val")

    # If checkpoint looks like a bare job_id (no slashes), resolve to actual path
    if "/" not in checkpoint and not checkpoint.startswith("microsoft"):
        checkpoint = str(shared / "jobs" / checkpoint / "checkpoints" / "best")

    _submit_eval_job(shared, dataset_id, checkpoint, split)
    return redirect(url_for("jobs.index"))
