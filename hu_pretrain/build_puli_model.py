#!/usr/bin/env python3
"""
Assemble a TrOCR-style VisionEncoderDecoder with:
  - Encoder : ViT from microsoft/trocr-base-handwritten  (image understanding)
  - Decoder : PULI-GPT-2 from NYTK/PULI-GPT-2            (Hungarian language prior)

Cross-attention layers are randomly initialised (expected — they bridge vision↔language).
Model, ViT image processor, and PULI tokenizer are all saved to --output so that
pretrain.py can load everything from a single directory.

Usage:
    python hu_pretrain/build_puli_model.py --output working/shared/puli_trocr_base

Then pre-train:
    python hu_pretrain/pretrain.py \\
        --base-model working/shared/puli_trocr_base \\
        --dataset  working/shared/datasets/synth_hu_100k \\
        --output   working/shared/pretrained_puli_v1 \\
        --epochs 15 --batch-size 8

License note: PULI-GPT-2 is cc-by-nc-4.0 (non-commercial use only).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    VisionEncoderDecoderModel,
)

ENCODER_DEFAULT = "microsoft/trocr-base-handwritten"
DECODER_DEFAULT = "NYTK/PULI-GPT-2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ViT + PULI-GPT-2 OCR model")
    parser.add_argument("--output", required=True, type=Path, help="Directory to save assembled model")
    parser.add_argument("--encoder", default=ENCODER_DEFAULT, help=f"HF encoder id (default: {ENCODER_DEFAULT})")
    parser.add_argument("--decoder", default=DECODER_DEFAULT, help=f"HF decoder id (default: {DECODER_DEFAULT})")
    args = parser.parse_args()

    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Encoder : {args.encoder}")
    print(f"Decoder : {args.decoder}")
    print(f"Output  : {out}")
    print()

    print("Loading tokenizer…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder)

    # GPT-2 has no pad token — use eos as pad (standard practice)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  pad_token set to eos_token")

    print("Loading image processor…", flush=True)
    image_processor = AutoImageProcessor.from_pretrained(args.encoder)

    print(f"Loading TrOCR encoder (extracting ViT from {args.encoder})…", flush=True)
    trocr_full = VisionEncoderDecoderModel.from_pretrained(args.encoder)
    vit_encoder = trocr_full.encoder  # ViT only — drop the RoBERTa decoder
    del trocr_full

    print(f"Loading PULI-GPT-2 decoder ({args.decoder})…", flush=True)
    puli_decoder = AutoModelForCausalLM.from_pretrained(args.decoder)

    print("Assembling VisionEncoderDecoderModel (cross-attention is randomly initialised)…", flush=True)
    model = VisionEncoderDecoderModel(encoder=vit_encoder, decoder=puli_decoder)

    # Required generation config — GPT-2 has no cls_token so use bos
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.use_cache = False
    model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.max_new_tokens = 128
    model.generation_config.num_beams = 1
    model.generation_config.early_stopping = False

    print("Saving model…", flush=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    image_processor.save_pretrained(out)

    print()
    print(f"Done. Saved to: {out}")
    print(f"  decoder_start_token_id : {model.config.decoder_start_token_id}")
    print(f"  pad_token_id           : {model.config.pad_token_id}")
    print(f"  vocab_size             : {model.config.vocab_size}")
    print()
    print("Next step:")
    print(f"  python hu_pretrain/pretrain.py --base-model {out} --dataset ... --output ...")


if __name__ == "__main__":
    main()
