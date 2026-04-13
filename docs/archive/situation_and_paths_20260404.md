# Inkwell — Situation Summary & Paths Forward
**Date:** 2026-04-04 (updated 2026-04-04)  
**Audience:** Human owner, or an LLM picking up this project for the first time.

---

## 1. What this project is

A private, personal digitisation project. The owner has **8 handwritten Hungarian personal diaries** (teen/young adult, written in Hungarian, estimated 1980s–1990s). The goal is to produce a machine-readable text transcript of the entire collection.

The owner built a full OCR pipeline from scratch called **Inkwell** to attempt this with an automated ML approach.

---

## 2. The corpus

| Item | Count |
|------|-------|
| Notebooks | 8 |
| Scanned pages | 798 |
| Segmented text lines | 15,997 |
| Manually transcribed lines | 1,035 (~6.5%) |
| Remaining unannotated lines | ~14,962 |

**Per-notebook line counts:**

| Notebook | Lines |
|----------|-------|
| Notebook 1 | 1,981 |
| Notebook 2 | 252 |
| Notebook 3 | 882 |
| Notebook 4 | 2,463 |
| Notebook 5 | 2,555 |
| Notebook 6 | 3,670 |
| Notebook 7 | 2,539 |
| Notebook 8 | 1,655 |

Average ~20 lines/page. The handwriting is **personal cursive Hungarian**, not printed text.

---

## 3. The Inkwell system (what was built)

A full human-in-the-loop OCR improvement pipeline:

**Stack:**
- Python, Flask web UI, SQLite database (`working/inkwell.db`)
- GPU worker system over a shared folder (dev machine + remote GPU server via NFS mount)
- Model: Microsoft TrOCR (`microsoft/trocr-base-handwritten`) fine-tuned via HuggingFace Transformers
- Remote GPU: personal server with RTX 2060 (6 GB VRAM)

**Key components:**
- `inkwell/web/` — Flask web app with annotation UI, job control, results viewer
- `scripts/finetune_trocr.py` — fine-tuning script (Seq2SeqTrainer)
- `scripts/eval_model.py` — per-line CER/WER evaluation with prediction output
- `scripts/export_gt.py` — exports annotated ground truth as train/val/test JSONL datasets
- `scripts/infer_unlabeled_pool.py` — runs inference on all ~15k unannotated lines
- `scripts/pick_next_samples.py` — active learning: picks best next lines to annotate
- `scripts/run_automation.py` — one-click: export → train → submit eval + pool inference
- `scripts/sync_code_to_gpu.py` — syncs local code to GPU server via rsync/SSH

**Web UI routes:**
- `/annotate` — queue-mode annotation with OCR suggestions displayed
- `/jobs` — automation control panel (run training, sync code, submit evals)
- `/jobs/results` — experiment and eval results overview (CER table by dataset/model/label)
- `/jobs/<job_id>/eval-detail` — per-line CER viewer with crop images, reference, and prediction

**Dataset/training loop:**
1. Human manually corrects OCR lines in web UI
2. Export GT → JSONL with configurable text-marker policy (`readable_text_v1`)
3. Fine-tune TrOCR from `microsoft/trocr-base-handwritten` base
4. Evaluate on frozen val/test split
5. Run inference on full 15k line pool
6. Generate next annotation suggestions (active learning ranking)
7. Repeat

---

## 4. Training results history

All runs used `microsoft/trocr-base-handwritten` unless noted. Metric is **Character Error Rate (CER)** on the fixed validation set — lower is better.

| Round | Dataset | Train samples | Val CER | Notes |
|-------|---------|---------------|---------|-------|
| 1 | gt_20260312_round1 | 133 | 0.702 | First fine-tune |
| 2 | gt_20260314_round2 | 319 | 0.604 | Policy-filtered |
| 3 | gt_20260317_round3 | 428 | 0.612 | — |
| 4 | gt_20260324_round4 | 523 | **0.5805** | Best result |
| 5 | gt_20260401_round5 | 692 | 0.5933 | +169 samples, no gain |
| 6 (large) | gt_20260401_round6 | 690 | 0.5929 | `trocr-large-handwritten` |
| 6 (stage1) | gt_20260401_round6 | 690 | 0.6444 | `trocr-base-stage1` |

**Baseline (no fine-tuning, `trocr-base-handwritten` on Hungarian):** CER ~0.76–0.79

---

## 5. The core problem

### 5.1 The model has plateaued

Despite going from 133 → 692 training samples, the best CER has only moved from 0.702 → 0.5805. The last two rounds (+169 samples each) produced **zero meaningful improvement**. The learning curve is flat.

### 5.2 Root cause: wrong base model for the language

`microsoft/trocr-base-handwritten` was pretrained on the **IAM English handwriting corpus**. Its decoder is a BERT model with a very strong **English language prior**.

When it encounters Hungarian words it cannot read confidently, it does not output phonetic noise — it outputs **English-looking word sequences** that match the visual stroke shapes. Examples from actual predictions:

| Reference (Hungarian) | Prediction (model output) |
|----------------------|--------------------------|
| garázsba néhány könyvért | garisella winding to request |
| az Ákosos fényképtekecset. | an Advisory Employment Assessment . |
| tudok 1 napnál továb | inside I reapplut trouble . |

This is not a training data size problem. It is a **pretrained model architecture mismatch**. The English prior is encoded deep in the model weights. 692 fine-tuning samples is nowhere near enough to override ~millions of English text tokens seen in pretraining.

### 5.3 `trocr-large-handwritten` did not help

More model capacity did not help (CER 0.5929 vs 0.5805 for base). This is expected: with only 690 training samples, the larger model has *more* English prior to fight against, not less.

### 5.4 `trocr-base-stage1` (no language prior) was worse

`stage1` starts from a generic visual encoder + empty decoder. Without any language model smoothing it produces phonetic garbage (repeated accented characters), CER 0.6444. Better for one class of errors, much worse overall.

### 5.5 Hardware constraint

RTX 2060, 6 GB VRAM. This rules out training or running models larger than ~base-sized TrOCR or similar. Fine-tuning a model from scratch is not feasible.

---

## 6. The annotation investment so far

1,035 lines manually corrected and stored in `inkwell.db`. These are:
- High-quality, immutable ground truth
- Split into train/val/test with frozen eval sets
- Useful as evaluation data regardless of what model is used next
- The work is not wasted — it will be needed to measure quality of any approach

---

## 7. External tool test results (tested 2026-04-04)

Before committing to further ML engineering, three off-the-shelf tools were tested on real pages/crops from the corpus.

### Transkribus — FAILED

- Uploaded a sample page to [transkribus.eu](https://www.transkribus.eu)
- Result: **completely unusable garbage** — random unrelated letters, no recognisable Hungarian words
- The pretrained Central European models did not handle this handwriting style at all
- **Verdict: abandoned**

### ChatGPT (GPT-4o vision) — FAILED

- Tested with individual line crops and full page images
- Result: recognised some isolated words correctly, but produced a mix of random Hungarian words and nonsense
- Output was not coherent or reconstructable — too many errors to be useful even as a correction base
- **Verdict: not usable for this handwriting**

### Claude (Anthropic) — Promising but insufficient

- Tested with full page images
- Result: **significantly better** — enough correct words that the original story could be partially reconstructed from the output
- Still many mistakes; would take nearly as long to correct as to transcribe manually
- **Verdict: best external tool tested, but not a practical solution at current quality**

### Conclusion

No external tool currently produces output good enough to accelerate transcription meaningfully for this specific handwriting. The handwriting is personal, idiosyncratic, and not well-represented in any pretrained model.

---

## 8. Current plan (decided 2026-04-04)

### Primary: Full manual transcription

The primary approach is **full manual transcription** using the Inkwell web annotation UI (`/annotate` queue mode). Treated as an enjoyable reading/transcribing project.

**Rationale:**
- No available tool is good enough for this handwriting
- The UI and workflow are already in place
- Result will be perfect — no ML errors, no hallucinations
- 1,035 lines already done (~6.5%); pace and tooling are established

**Volume estimate:**
- ~14,962 lines remaining
- At ~25–35 lines/minute for familiar handwriting: **~65–80 hours total**
- At 5 hours/week: ~3–4 months

### Secondary: Background ML improvements

In parallel with manual transcription, the following ML directions are worth exploring when motivated. They are not blockers for transcription work.

#### ML Goal 1: Better line segmentation

Known weaknesses with the current CV projection-based segmenter:
- Clips top accents (á, é, ő, ű, etc.) on tight crops
- Occasionally merges adjacent lines on dense pages
- Produces partial crops for margin notes or indented lines

**Plan:**
- Improve segmentation algorithm or crop margin parameters
- **Critical constraint: preserve existing `line_id` mappings** — the 1,035 annotated lines must stay matched to their correct crops. Re-segmentation must only apply to unannotated lines, or use a new segmentation profile that coexists with existing ones
- Tool already exists: `scripts/recrop_lines.py` for safe re-cropping with ID preservation
- Better crops → better model input → better CER, regardless of model

#### ML Goal 2: Hungarian training data

The English language prior in TrOCR is the root cause of hallucinated English words (see section 5). Two approaches to address it:

**2a. Use existing Hungarian HTR/OCR datasets**
- Search HuggingFace Datasets, Papers With Code, and academic sources for Hungarian handwriting or OCR corpora
- Even printed Hungarian text data helps — it teaches the decoder Hungarian vocabulary and orthography before fine-tuning on real handwriting
- Pre-fine-tune a base model on this, then fine-tune again on the real diary annotations

**2b. Generate synthetic Hungarian handwriting data**
- Take a large Hungarian text corpus (e.g. Project Gutenberg Hungarian books, Hungarian Wikipedia, Common Crawl Hungarian subset)
- Render text as line images using varied handwriting-style fonts with augmentation (rotation, noise, blur, contrast)
- Use as pre-training data to override the English decoder prior
- Libraries: `Pillow`, `trdg` (TextRecognitionDataGenerator)
- Standard approach in low-resource historical OCR literature

**Estimated effort for synthetic data path:** 1–2 weeks of engineering  
**Estimated gain:** Unknown — could push CER below 0.40 if synthetic quality is good; may not help if font/real handwriting gap is too large

**Note:** Both ML goals should be pursued without disturbing the annotation DB or existing line IDs. The 1,035 annotations are permanent ground truth and must never be overwritten.

---

## 9. What's already useful regardless of approach

- **1,035 annotated lines** — high-quality evaluation set; can measure any new model's quality immediately
- **The Inkwell eval tooling** — `scripts/eval_model.py` + the `/jobs/eval-detail` web UI — can evaluate any checkpoint or model output against the frozen ground truth
- **The web annotation UI** — `/annotate` is the primary transcription tool; queue mode + OCR suggestion display works well
- **The segmentation** — all 15,997 lines are already cropped into individual images; these can be fed to any vision API or model without re-doing segmentation

---

## 10. Key files and locations

```
inkwell/                        # Main Python package
scripts/                        # All runnable scripts
working/inkwell.db              # SQLite DB (annotations, lines, pages, notebooks)
working/shared/                 # Shared folder between dev machine and GPU server
  jobs/                         # Job queue (finetune, eval, infer_pool)
  datasets/                     # Exported GT datasets (train/val/test JSONL + crop images)
  suggestions/                  # Active learning suggestion files
automation.toml                 # Main config (GPU host, dataset ID, job params)
docs/inkwell_plan_final.md      # Architecture and design decisions
docs/runbook.md                 # Step-by-step operational procedures
docs/session_handoff.md         # Quick-start for new sessions
```

---

## 11. Context for an LLM resuming this project

**Current primary activity (as of 2026-04-04):** Manual transcription via `/annotate` queue mode. The owner is working through notebooks line by line as a personal reading/transcription project.

**Active background ML directions:**
1. Improve line segmentation without breaking existing `line_id` mappings
2. Find or generate Hungarian text/handwriting training data to reduce the English language prior in TrOCR

**Do not restart full ML training loops** without first checking whether segmentation or training data improvements are in place — the current CER ceiling (~0.58) cannot be broken with the current setup alone.

**General conventions:**
- The owner is an experienced developer who wants to understand what's happening, not just get magic outputs
- Prefer small focused changes over large rewrites
- The annotation data in `inkwell.db` is precious — never overwrite immutable human labels
- The `line_id` is the authoritative identifier; never change the mapping
- The GPU server is `hunbug@NeoLinux`; code syncs via `scripts/sync_code_to_gpu.py`
- The web server runs locally on the dev machine at `http://127.0.0.1:5000`
- All experiments are tracked in the shared jobs folder visible to both machines via NFS mount
