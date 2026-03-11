# Remote Testing Guide (Dev machine + GPU machine)

This guide explains exactly how to test the automation loop with your current setup:
- Dev machine runs launcher and web UI
- GPU machine runs worker + training/eval
- Shared network folder is the communication channel

---

## 1) Architecture (what talks to what)

### Shared-folder mode (no direct rsync needed)
- Both machines see the same folder: [automation.toml](../automation.toml) `shared.path`
- Dev machine writes:
  - `datasets/...` (GT export)
  - `jobs/.../job.json` (new task)
- GPU worker reads jobs, writes:
  - `jobs/.../progress.json`
  - `jobs/.../result.json`
  - `worker_status.json`
- Web `/jobs` reads those files.

In this mode, `gpu.user` is not required for job flow itself.

Important nuance:
- the **absolute path may differ** between machines,
- e.g. dev machine: `/home/akoss/.../inkwell-automation`
- GPU machine: `/home/hunbug/.../inkwell-automation`
- that is OK, as long as both point to the same shared content.
- on the GPU machine, set `INKWELL_SHARED` to the server-side path.

### Optional GPU-cache mode (faster local I/O on GPU)
- Dev launcher additionally `rsync`s dataset to GPU local path (`remote_dataset_cache_root`)
- This needs SSH access (`gpu.user`, passkey)
- Worker can use local cache for better performance.

---

## 2) Current config meaning (important clarification)

In your config:
- `shared.path = "/home/akoss/mnt/lara-playground/playground/inkwell-automation"`
  - This is the primary communication path and dataset location.
- `gpu.user = "hunbug"`
  - Needed only when SSH actions are used (optional rsync cache sync).
- `remote_dataset_cache_root = "~/work/inkwell-datasets"`
  - Path on GPU machine used only for optional rsync cache copy.

So your understanding is correct: shared folder is enough for the core workflow.

---

## 3) New safety precheck now in launcher

`scripts/run_automation.py` now checks before doing anything:
1. Shared **base path** is accessible:
   - `/home/akoss/mnt/lara-playground/playground`
2. Shared folder is readable/writable
3. Temporary write test succeeds
4. Then normal checks continue (ping, active jobs, etc.)

This avoids accidental local writes when the network mount is missing.

---

## 4) One-time setup checklist

### On dev machine
1. Activate venv
2. Fill [automation.toml](../automation.toml)
3. Decide mode:
   - shared-only: `sync_to_gpu_cache = false`
   - cache-sync: `sync_to_gpu_cache = true`

### On GPU machine
1. Ensure CUDA works (`nvidia-smi`)
2. Install Python deps for training (`requirements-ml.txt` + project deps)
3. Set `INKWELL_SHARED` to the shared folder path **as seen on the GPU machine**
4. Start worker:
   - `python scripts/gpu_worker.py`
   - or with local cache path: `python scripts/gpu_worker.py --local-datasets /path/to/local/datasets`

---

## 5) Remote test procedure (first run)

### Step A — worker alive check
On GPU machine:
- start worker and keep it running (tmux/screen recommended)

On dev machine:
- open `/jobs` page and verify Worker status appears (`idle`/`running`)

### Step B — dry automation run
On dev machine:
- `python scripts/run_automation.py`

Expected behavior:
1. prechecks pass (shared path + ping + no active jobs)
2. incremental split freeze runs
3. dataset reused/exported
4. optional rsync happens (only if enabled)
5. job is submitted

### Step C — monitor progress
- Open `/jobs`
- Confirm transitions: `pending -> running -> completed` (or failed)
- Check metrics and log tail there

### Step D — cancellation test
- In `/jobs`, click cancel for running job
- or create `CANCEL` file in job folder
- worker should stop and mark job `cancelled`

### Step E — continuation test
- Submit next finetune with `resume_from` checkpoint path in config (`[job].resume_from`)
- run launcher again

---

## 6) Minimal command set

### Dev machine
- `python scripts/run_automation.py`

### GPU machine
- `export INKWELL_SHARED="/home/akoss/mnt/lara-playground/playground/inkwell-automation"`
- `python scripts/gpu_worker.py`

---

## 7) Troubleshooting quick map

- **"Shared base path does not exist"**
  - Network mount is not available on dev machine.
- **Ping fails**
  - Hostname/IP issue or machine offline.
- **No worker heartbeat on `/jobs`**
  - Worker not running or using wrong `INKWELL_SHARED`.
- **Job stays `pending` forever**
  - Worker not seeing same shared folder path.
- **rsync fails**
  - SSH/passkey/user/path mismatch; disable cache sync to continue in shared-only mode.

---

## 8) Recommended default for your current setup

Since the shared folder is visible from both machines:
- keep `sync_to_gpu_cache = false` initially
- rely on shared-folder workflow first
- enable cache sync later only if you measure I/O bottlenecks
