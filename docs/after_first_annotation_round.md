# After the First Annotation Round

This is the exact workflow to follow after you finish a round of annotation.

The goal is to answer:
- what to click on the web page,
- what to run in Konsole,
- when `rsync` happens automatically,
- and when you need to do something manually.

---

## Short answer first

### Do I need to manually `rsync` the dataset to the GPU server?

**No, normally not.**

With the current setup:
- the dataset is exported into the **shared folder**,
- the GPU machine sees that shared folder directly,
- and the GPU worker can also **auto-copy** the dataset into its local cache.

So for the **dataset**, you usually do **not** run `rsync` manually.

### When is `rsync` used then?

There are two separate things:

1. **Dataset export / availability**
   - handled by `python scripts/run_automation.py`
   - exported into the shared folder
   - no manual `rsync` needed

2. **Code sync to GPU server**
   - only needed if your Python/scripts/templates changed
   - handled by:
     - web button: **Sync code + restart worker**
     - or command: `python scripts/sync_code_to_gpu.py --force --start-runner`

So:
- **dataset sync:** automatic through shared folder / worker cache
- **code sync:** done by the sync script/button

---

## The normal loop after annotation

The normal loop is:

1. annotate lines
2. decide the next dataset version
3. export / submit through automation
4. GPU worker trains/evaluates
5. inspect results in `/jobs`
6. analyze errors
7. annotate next batch

---

## Recommended workflow (web-first)

This is the easiest version.

### Step 0 — finish your annotation round

Example: you reach ~200 corrected lines.

---

### Step 1 — set the next dataset version

Open [automation.toml](../automation.toml).

Update:
- `[dataset].dataset_id`

Example:

- from: `gt_20260311`
- to: `gt_20260312_round1`

Recommended:
- use a **new dataset id** for milestone runs
- this keeps runs reproducible

If you really want to overwrite the same dataset folder instead, set:
- `[dataset].force_reexport = true`

But usually a new dataset id is better.

---

### Step 2 — if code changed, sync code to GPU

You only need this if you changed code locally since the last GPU run.

Examples of code changes:
- `scripts/*.py`
- web routes/templates
- training logic
- worker logic
- evaluation logic

On the web page:
- open `/jobs`
- click **Sync code + restart worker**

What it does:
- checks the remote worker
- stops it if needed
- syncs repo code to the GPU machine
- starts the worker again
- shows output in the sync log block on the jobs page

If you did **not** change code, skip this step.

---

### Step 3 — export dataset and submit the job

On the web page:
- open `/jobs`
- click **Run automation now**

What it does:
- runs prechecks
- freezes new split assignments incrementally if needed
- exports the dataset into the shared folder
- optionally syncs dataset cache if configured
- submits the finetune/eval job

You do **not** need to manually export and manually copy the dataset in the normal flow.

---

### Step 4 — monitor progress

Still on `/jobs`:
- watch worker status
- watch job status (`pending`, `running`, `completed`, `failed`)
- open the latest log block
- if needed, cancel from the UI

---

### Step 5 — evaluate and compare

After the run finishes, compare:
- baseline model vs
- fine-tuned model

Use `/jobs` to inspect:
- loss
- CER / WER
- log output

Then do error analysis.

---

## Recommended workflow (Konsole / CLI)

Use this if you prefer commands.

### Step 0 — activate the virtual environment

From the repo root:

`source .venv/bin/activate`

---

### Step 1 — edit dataset version

Open [automation.toml](../automation.toml) and update:
- `[dataset].dataset_id`

Example:
- `gt_20260312_round1`

---

### Step 2 — if code changed, sync code and restart worker

Run:

`python scripts/sync_code_to_gpu.py --force --start-runner`

Use this when:
- training code changed
- worker code changed
- evaluation code changed
- job/orchestration code changed

If code did not change, skip this step.

---

### Step 3 — run automation

Run:

`python scripts/run_automation.py`

This does the normal pipeline:
- prechecks
- split freeze
- dataset export
- optional dataset cache sync
- job submission

---

### Step 4 — inspect progress in the browser

Open:
- `/jobs`

That is the easiest place to monitor the run.

---

## Exact “usual case” after annotation

This is probably the flow you want most of the time.

### Case A — only annotations changed

If you only added/corrected annotations, and code did not change:

#### Web
1. update [automation.toml](../automation.toml) dataset id
2. open `/jobs`
3. click **Run automation now**
4. watch progress

#### Konsole
1. `source .venv/bin/activate`
2. edit [automation.toml](../automation.toml)
3. `python scripts/run_automation.py`
4. watch `/jobs`

In this case:
- **no manual `rsync` needed**
- **no code sync needed**

---

### Case B — annotations changed and code changed too

If you changed annotation data **and** changed scripts/code:

#### Web
1. update [automation.toml](../automation.toml) dataset id
2. open `/jobs`
3. click **Sync code + restart worker**
4. wait until sync finishes
5. click **Run automation now**
6. watch progress

#### Konsole
1. `source .venv/bin/activate`
2. edit [automation.toml](../automation.toml)
3. `python scripts/sync_code_to_gpu.py --force --start-runner`
4. `python scripts/run_automation.py`
5. watch `/jobs`

---

## What gets written where

After automation runs:

### Dataset export
The dataset is written under the shared folder:
- `shared/datasets/<dataset_id>/`

That contains:
- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `crops/`
- `manifest.json`

### Job state
Jobs are written under:
- `shared/jobs/<job_id>/`

That contains things like:
- `job.json`
- `progress.json`
- `worker.log`
- `result.json`
- `checkpoints/`

### Worker heartbeat
The worker writes:
- `shared/worker_status.json`

---

## What is automatic vs manual

### Automatic
- exporting dataset to shared folder
- worker seeing jobs from shared folder
- worker updating progress/result files
- optional local GPU dataset cache copy

### Manual
- changing `dataset_id` when you want a new milestone dataset
- syncing code to GPU **if code changed**
- starting automation for the next run
- checking results and deciding next iteration

---

## Suggested milestone routine

For your current stage:

1. annotate until the round is done
2. set a new dataset id
3. if code changed: sync code + restart worker
4. run automation
5. wait for training/eval
6. compare results
7. inspect bad predictions
8. annotate the next batch

---

## Minimal operator checklist

### If only annotation changed

- update [automation.toml](../automation.toml)
- run `python scripts/run_automation.py`
- monitor `/jobs`

### If code changed too

- update [automation.toml](../automation.toml)
- run `python scripts/sync_code_to_gpu.py --force --start-runner`
- run `python scripts/run_automation.py`
- monitor `/jobs`

### Web equivalent

- **Sync code + restart worker** (only if code changed)
- **Run automation now**
- watch `/jobs`

---

## One-line mental model

After annotation:
- **new data only** -> run automation
- **new data + new code** -> sync code, then run automation

That is the loop.
