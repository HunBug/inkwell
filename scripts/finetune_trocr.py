#!/usr/bin/env python3
"""
Fine-tune TrOCR on the exported GT dataset.

Called by gpu_worker.py, but can also be run standalone.

Usage:
    python scripts/finetune_trocr.py \\
        --dataset /shared/datasets/gt_20260311 \\
        --job-dir /shared/jobs/finetune_20260311_143022 \\
        --base-model microsoft/trocr-base-handwritten \\
        --epochs 10 --batch-size 8 --lr 5e-5

Writes:
  job-dir/checkpoints/best/    ← best val-loss checkpoint
  job-dir/checkpoints/last/    ← final epoch checkpoint
  job-dir/result.json
  (progress.json is updated throughout)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LineOCRDataset(Dataset):
    def __init__(self, jsonl_path: Path, processor: TrOCRProcessor, crops_dir: Path) -> None:
        self.items: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))
        self.processor = processor
        self.crops_dir = crops_dir

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        image_rel = Path(item["image"])
        if image_rel.parts and image_rel.parts[0] == "crops":
            image_path = self.crops_dir.parent / image_rel
        else:
            image_path = self.crops_dir / image_rel
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            item["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        ).input_ids.squeeze(0)
        # Mask padding tokens so they don't contribute to loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# Progress + cancel callback
# ---------------------------------------------------------------------------

class JobProgressCallback(TrainerCallback):
    def __init__(self, job_dir: Path, total_epochs: int) -> None:
        self.job_dir = job_dir
        self.total_epochs = total_epochs
        self._last_train_loss: float | None = None

    def _write(self, **kwargs) -> None:
        p = self.job_dir / "progress.json"
        existing: dict = {}
        if p.exists():
            try:
                existing = json.loads(p.read_text())
            except Exception:
                pass
        existing.update(kwargs)
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        p.write_text(json.dumps(existing, indent=2))

    def _check_cancel(self, control: TrainerControl) -> None:
        if (self.job_dir / "CANCEL").exists():
            print("\n[trainer] CANCEL file detected — stopping training.", flush=True)
            control.should_training_stop = True

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._check_cancel(control)
        if state.log_history:
            last = state.log_history[-1]
            if "loss" in last:
                self._last_train_loss = last["loss"]

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._check_cancel(control)
        epoch = int(state.epoch or 0)
        self._write(
            status="running",
            epoch=epoch,
            total_epochs=self.total_epochs,
            step=state.global_step,
            train_loss=self._last_train_loss,
            message=f"Epoch {epoch}/{self.total_epochs} complete",
        )

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics:
            val_loss = metrics.get("eval_loss")
            self._write(val_loss=val_loss, message=f"Eval loss: {val_loss:.4f}" if val_loss else "Evaluating...")


# ---------------------------------------------------------------------------
# CER helper (no extra deps)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            tmp = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def compute_cer(predictions: list[str], references: list[str]) -> float:
    total_dist = sum(_edit_distance(p, r) for p, r in zip(predictions, references))
    total_len = sum(len(r) for r in references)
    return total_dist / total_len if total_len > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR")
    parser.add_argument("--dataset", required=True, help="Dataset directory (from export_gt.py)")
    parser.add_argument("--job-dir", required=True, help="Job directory for progress/checkpoints")
    parser.add_argument("--base-model", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--resume-from", default=None, help="Resume from this checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    job_dir = Path(args.job_dir)
    checkpoints_dir = job_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def write_progress(**kwargs):
        p = job_dir / "progress.json"
        existing: dict = {}
        if p.exists():
            try:
                existing = json.loads(p.read_text())
            except Exception:
                pass
        existing.update(kwargs)
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        p.write_text(json.dumps(existing, indent=2))

    write_progress(status="running", message="Loading model...")
    print(f"Loading model: {args.base_model}", flush=True)

    model_path = args.resume_from or args.base_model
    processor = TrOCRProcessor.from_pretrained(args.base_model, use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    # Required config for TrOCR generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.use_cache = False

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    crops_dir = dataset_dir / "crops"
    train_ds = LineOCRDataset(dataset_dir / "train.jsonl", processor, crops_dir)
    val_ds = LineOCRDataset(dataset_dir / "val.jsonl", processor, crops_dir)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}", flush=True)
    write_progress(message=f"Loaded: train={len(train_ds)} val={len(val_ds)}")

    def build_training_args(current_batch_size: int) -> Seq2SeqTrainingArguments:
        return Seq2SeqTrainingArguments(
            output_dir=str(checkpoints_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=current_batch_size,
            per_device_eval_batch_size=max(1, current_batch_size),
            gradient_accumulation_steps=max(1, args.gradient_accumulation),
            learning_rate=args.lr,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            predict_with_generate=False,
            logging_steps=10,
            fp16=torch.cuda.is_available(),
            report_to="none",  # no wandb/tensorboard by default
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            optim="adafactor",
        )

    train_result = None
    current_batch_size = max(1, args.batch_size)
    max_oom_retries = 4
    oom_retry = 0

    while True:
        write_progress(
            status="running",
            message=(
                f"Training start (batch={current_batch_size}, "
                f"grad_accum={max(1, args.gradient_accumulation)})"
            ),
            batch_size=current_batch_size,
            gradient_accumulation=max(1, args.gradient_accumulation),
            oom_retries=oom_retry,
        )

        training_args = build_training_args(current_batch_size)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            callbacks=[JobProgressCallback(job_dir, args.epochs)],
        )

        try:
            train_result = trainer.train()
            break
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if current_batch_size <= 1 or oom_retry >= max_oom_retries:
                raise

            oom_retry += 1
            current_batch_size = max(1, current_batch_size // 2)
            print(
                f"[OOM] Retrying with smaller batch size: {current_batch_size}",
                flush=True,
            )
            write_progress(
                status="running",
                message=f"OOM encountered, retrying with batch={current_batch_size}",
                batch_size=current_batch_size,
                oom_retries=oom_retry,
            )
            continue
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" not in msg and "cuda" not in msg:
                raise

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if current_batch_size <= 1 or oom_retry >= max_oom_retries:
                raise

            oom_retry += 1
            current_batch_size = max(1, current_batch_size // 2)
            print(
                f"[OOM-like RuntimeError] Retrying with smaller batch size: {current_batch_size}",
                flush=True,
            )
            write_progress(
                status="running",
                message=f"OOM-like error, retrying with batch={current_batch_size}",
                batch_size=current_batch_size,
                oom_retries=oom_retry,
            )
            continue

    assert train_result is not None

    # Save best model to checkpoints/best
    best_dir = checkpoints_dir / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    print(f"Best model saved to {best_dir}", flush=True)

    # Final eval + CER on val set
    write_progress(status="running", message="Final evaluation...")
    val_predictions = []
    val_references = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for item_dict in val_ds:
            pixel_values = item_dict["pixel_values"].unsqueeze(0).to(device)
            generated_ids = model.generate(pixel_values)
            pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            # Recover reference text from dataset
            val_predictions.append(pred)

    # Re-read references from jsonl directly (labels were masked)
    with open(dataset_dir / "val.jsonl", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            val_references.append(item["text"])

    val_cer = compute_cer(val_predictions, val_references)
    print(f"Final val CER: {val_cer:.4f}", flush=True)

    result = {
        "status": "completed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "resume_from": args.resume_from,
        "epochs": args.epochs,
        "batch_size": current_batch_size,
        "gradient_accumulation": max(1, args.gradient_accumulation),
        "oom_retries": oom_retry,
        "train_runtime_secs": train_result.metrics.get("train_runtime"),
        "final_train_loss": train_result.metrics.get("train_loss"),
        "final_val_cer": val_cer,
        "best_checkpoint": str(best_dir),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    (job_dir / "result.json").write_text(json.dumps(result, indent=2))
    write_progress(
        status="completed",
        val_cer=val_cer,
        batch_size=current_batch_size,
        message=f"Done. Val CER: {val_cer:.4f}",
    )


if __name__ == "__main__":
    main()
