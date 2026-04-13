#!/usr/bin/env python3
"""
Pre-train TrOCR on a synthetic Hungarian dataset (e.g. synth_hu_20k from download_data.py).
Produces a checkpoint with a Hungarian-biased decoder that can then be fine-tuned on
real diary annotations using the existing scripts/finetune_trocr.py.

MLflow tracking is written to --tracking-uri (default: working/shared/mlflow/).
Run the UI on your dev machine:
    mlflow ui --backend-store-uri /path/to/working/shared/mlflow

Usage:
    python hu_pretrain/pretrain.py \\
        --dataset working/shared/datasets/synth_hu_20k \\
        --output working/shared/pretrained_hu \\
        --epochs 3 \\
        --batch-size 8

    # With explicit tracking URI:
    python hu_pretrain/pretrain.py \\
        --dataset working/shared/datasets/synth_hu_20k \\
        --output working/shared/pretrained_hu \\
        --tracking-uri working/shared/mlflow
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
    AutoImageProcessor,
    AutoTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)


def _load_image_processor_and_tokenizer(model_id: str) -> tuple:
    """Load image processor + tokenizer for any supported model.

    Returns (image_processor, tokenizer).  Works for both:
    - Standard TrOCR checkpoints (microsoft/trocr-*)
    - PULI-based assembled checkpoints (ViT encoder + GPT-2 decoder)
    """
    try:
        proc = TrOCRProcessor.from_pretrained(model_id, use_fast=True)
        return proc.image_processor, proc.tokenizer
    except Exception:
        return (
            AutoImageProcessor.from_pretrained(model_id),
            AutoTokenizer.from_pretrained(model_id),
        )

try:
    import mlflow
except ImportError:
    sys.exit("mlflow not installed. Run: pip install mlflow>=2.10")


# ---------------------------------------------------------------------------
# Dataset  (same as finetune_trocr.py — intentionally duplicated, not imported)
# ---------------------------------------------------------------------------

class LineOCRDataset(Dataset):
    def __init__(self, jsonl_path: Path, image_processor, tokenizer, crops_dir: Path) -> None:
        self.items: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))
        self.image_processor = image_processor
        self.tokenizer = tokenizer
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
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.tokenizer(
            item["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        ).input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# CER helper
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
# MLflow callback
# ---------------------------------------------------------------------------

class MLflowCallback(TrainerCallback):
    """Logs train loss per step and eval loss per epoch to the active MLflow run."""

    def __init__(self, total_epochs: int) -> None:
        self.total_epochs = total_epochs
        self._last_train_loss: float | None = None

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if state.log_history:
            last = state.log_history[-1]
            if "loss" in last:
                self._last_train_loss = last["loss"]
                mlflow.log_metric("train_loss_step", last["loss"], step=state.global_step)

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        epoch = int(state.epoch or 0)
        # Fall back to log_history if on_step_end never fired (tiny dataset / large logging_steps)
        if self._last_train_loss is None:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    self._last_train_loss = entry["loss"]
                    break
        if self._last_train_loss is not None:
            mlflow.log_metric("train_loss_epoch", self._last_train_loss, step=epoch)
        print(f"[mlflow] epoch {epoch}/{self.total_epochs}  train_loss={self._last_train_loss}", flush=True)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs) -> None:
        if metrics and "eval_loss" in metrics:
            epoch = int(state.epoch or 0)
            mlflow.log_metric("val_loss", metrics["eval_loss"], step=epoch)


class ProcessorSaveCallback(TrainerCallback):
    """Saves image_processor + tokenizer into every epoch checkpoint so the dir is self-contained."""

    def __init__(self, image_processor, tokenizer, checkpoints_dir: Path) -> None:
        self._image_processor = image_processor
        self._tokenizer = tokenizer
        self._checkpoints_dir = checkpoints_dir

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        ckpt_dir = self._checkpoints_dir / f"checkpoint-{state.global_step}"
        if ckpt_dir.exists():
            self._image_processor.save_pretrained(str(ckpt_dir))
            self._tokenizer.save_pretrained(str(ckpt_dir))


# ---------------------------------------------------------------------------
# Val CER pass
# ---------------------------------------------------------------------------

def run_val_cer(
    model: VisionEncoderDecoderModel,
    image_processor,
    tokenizer,
    val_jsonl: Path,
    crops_dir: Path,
    max_new_tokens: int,
    device: torch.device,
    limit: int = 200,
) -> float:
    """Quick CER pass on at most `limit` val examples."""
    model.eval()
    preds: list[str] = []
    refs: list[str] = []

    with open(val_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            item = json.loads(line.strip())
            text = item["text"]
            image_rel = Path(item["image"])
            if image_rel.parts and image_rel.parts[0] == "crops":
                image_path = crops_dir.parent / image_rel
            else:
                image_path = crops_dir / image_rel
            image = Image.open(image_path).convert("RGB")
            pixel_values = (
                image_processor(images=image, return_tensors="pt").pixel_values.to(device)
            )
            with torch.no_grad():
                ids = model.generate(pixel_values, max_new_tokens=max_new_tokens)
            pred = tokenizer.decode(ids[0], skip_special_tokens=True)
            preds.append(pred)
            refs.append(text)

    return compute_cer(preds, refs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train TrOCR on synthetic Hungarian data")
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset dir (train.jsonl + val.jsonl + crops/)")
    parser.add_argument("--output", required=True, type=Path, help="Output dir for checkpoints and result.json")
    parser.add_argument("--base-model", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--tracking-uri",
        type=Path,
        default=None,
        help="MLflow tracking URI (default: <repo-root>/working/shared/mlflow)",
    )
    parser.add_argument("--run-name", default=None, help="MLflow run name (auto-generated if omitted)")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Path to a checkpoint directory to resume/warm-start from. "
            "If the dir contains trainer_state.json (saved by a previous pretrain.py run), "
            "optimizer + scheduler are fully restored. "
            "Otherwise (e.g. a 'best/' dir), model weights are loaded and optimizer starts fresh."
        ),
    )
    args = parser.parse_args()

    dataset_dir = args.dataset.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    for required in [dataset_dir / "train.jsonl", dataset_dir / "val.jsonl"]:
        if not required.exists():
            sys.exit(f"Required file not found: {required}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # MLflow setup
    if args.tracking_uri is not None:
        tracking_uri = args.tracking_uri.expanduser().resolve()
    else:
        # Default: <repo-root>/working/shared/mlflow
        tracking_uri = Path(__file__).resolve().parents[1] / "working" / "shared" / "mlflow"
    tracking_uri.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri.as_uri())
    mlflow.set_experiment("hu_pretrain")

    run_name = args.run_name or f"pretrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    print(f"MLflow tracking URI : {tracking_uri}")
    print(f"Run name            : {run_name}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "base_model": args.base_model,
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "dataset": dataset_dir.name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "lr": args.lr,
            "max_new_tokens": args.max_new_tokens,
        })

        # Log manifest for traceability
        manifest_path = dataset_dir / "manifest.json"
        if manifest_path.exists():
            mlflow.log_artifact(str(manifest_path), artifact_path="dataset")

        resume_path: Path | None = args.resume_from.expanduser().resolve() if args.resume_from else None
        base_to_load = str(resume_path) if resume_path else args.base_model
        warm_start = resume_path is not None and not (resume_path / "trainer_state.json").exists()
        full_resume = resume_path is not None and (resume_path / "trainer_state.json").exists()

        print(f"Loading model: {base_to_load}", flush=True)
        if warm_start:
            print("  (warm start — model weights restored, optimizer starts fresh)", flush=True)
        elif full_resume:
            print("  (full resume — model + optimizer + scheduler restored)", flush=True)
        image_processor, tokenizer = _load_image_processor_and_tokenizer(base_to_load)
        model = VisionEncoderDecoderModel.from_pretrained(base_to_load)

        # cls_token_id exists in TrOCR (RoBERTa decoder) but not in GPT-2 decoders
        decoder_start = getattr(tokenizer, "cls_token_id", None) or tokenizer.bos_token_id
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        model.config.decoder_start_token_id = decoder_start
        model.config.pad_token_id = pad
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.use_cache = False
        model.generation_config.max_new_tokens = max(8, args.max_new_tokens)
        model.generation_config.num_beams = 1
        model.generation_config.early_stopping = False

        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

        crops_dir = dataset_dir / "crops"
        train_ds = LineOCRDataset(dataset_dir / "train.jsonl", image_processor, tokenizer, crops_dir)
        val_ds = LineOCRDataset(dataset_dir / "val.jsonl", image_processor, tokenizer, crops_dir)
        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}", flush=True)
        mlflow.log_params({"train_samples": len(train_ds), "val_samples": len(val_ds)})

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(checkpoints_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=max(1, args.batch_size),
            per_device_eval_batch_size=max(1, args.batch_size),
            gradient_accumulation_steps=max(1, args.gradient_accumulation),
            learning_rate=args.lr,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=False,
            predict_with_generate=False,
            logging_steps=10,
            fp16=torch.cuda.is_available(),
            report_to="none",
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            optim="adafactor",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            callbacks=[
                MLflowCallback(args.epochs),
                ProcessorSaveCallback(image_processor, tokenizer, checkpoints_dir),
            ],
        )

        print("Training…", flush=True)
        resume_arg = str(resume_path) if full_resume else None
        trainer.train(resume_from_checkpoint=resume_arg)

        # Save final checkpoint
        best_dir = output_dir / "checkpoints" / "best"
        model.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)
        print(f"Checkpoint saved: {best_dir}", flush=True)
        mlflow.log_artifact(str(best_dir), artifact_path="checkpoint")

        # Final CER on val (capped at 200 examples)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        val_cer = run_val_cer(
            model, image_processor, tokenizer,
            dataset_dir / "val.jsonl", crops_dir,
            max_new_tokens=args.max_new_tokens,
            device=device,
            limit=200,
        )
        print(f"Final val CER (first 200): {val_cer:.4f}", flush=True)
        mlflow.log_metric("final_val_cer", val_cer)

        result = {
            "run_name": run_name,
            "base_model": args.base_model,
            "dataset": dataset_dir.name,
            "epochs": args.epochs,
            "final_val_cer": val_cer,
            "checkpoint": str(best_dir),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2))
        mlflow.log_artifact(str(result_path))
        print(f"\nResult: {result_path}")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
