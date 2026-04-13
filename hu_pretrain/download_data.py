#!/usr/bin/env python3
"""
Download a subset of the HuHTR synthetic Hungarian dataset from HuggingFace
and save it in the same JSONL+crops format used by the rest of the inkwell pipeline.

Source: AlhitawiMohammed22/lines_hu_v2_1  (Apache 2.0)
  Subset used: lines_hu_v7 — ~100k Hungarian-text image/text pairs, printed fonts
  Files:
    CollectedFrom/lines_hu_v7/labels.parquet  {file_name, text}
    CollectedFrom/lines_hu_v7/images.zip      JPEG images by name

NOTE: The streaming/load_dataset API for this repo hits corrupt JSONL shards.
We bypass that by using hf_hub_download to fetch the parquet labels + images.zip
directly, which are clean.

Output layout (mirrors existing GT datasets):
  <output_dir>/
    train.jsonl      {image: "crops/<name>", text: "..."}
    val.jsonl
    manifest.json
    crops/           JPEG images

Usage:
    python hu_pretrain/download_data.py \\
        --output working/shared/datasets/synth_hu_20k \\
        --count 20000 \\
        --val-ratio 0.05

    # Smoke test (fast — only extracts the first N images):
    python hu_pretrain/download_data.py \\
        --output /tmp/synth_hu_smoke \\
        --count 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


_DATASET_REPO = "AlhitawiMohammed22/lines_hu_v2_1"
_LABELS_FILE = "CollectedFrom/lines_hu_v7/labels.parquet"
_IMAGES_FILE = "CollectedFrom/lines_hu_v7/images.zip"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def download(output_dir: Path, total_count: int, val_ratio: float, seed: int = 42) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install huggingface-hub")

    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas not installed. Run: pip install pandas pyarrow")

    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Download labels parquet (~5 MB) ---
    print(f"Downloading labels from {_DATASET_REPO} …", flush=True)
    labels_path = hf_hub_download(
        _DATASET_REPO,
        repo_type="dataset",
        filename=_LABELS_FILE,
    )
    df = pd.read_parquet(labels_path)
    print(f"  Labels loaded: {len(df)} rows")

    if len(df) == 0:
        sys.exit("Labels parquet is empty — check dataset access.")

    # Sample up to total_count rows (reproducible)
    if total_count < len(df):
        df = df.sample(n=total_count, random_state=seed).reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        total_count = len(df)

    # --- Step 2: Download images.zip (~400-600 MB, cached after first run) ---
    print(f"Downloading images zip from {_DATASET_REPO} (cached after first run) …", flush=True)
    images_zip_path = hf_hub_download(
        _DATASET_REPO,
        repo_type="dataset",
        filename=_IMAGES_FILE,
    )
    print(f"  Images zip ready: {images_zip_path}")

    # Build a quick lookup: filename → row in df
    wanted: dict[str, str] = {
        str(row["file_name"]): str(row["text"])
        for _, row in df.iterrows()
    }

    # --- Step 3: Extract only the needed images ---
    print(f"Extracting {len(wanted)} images …", flush=True)
    extracted = 0
    with zipfile.ZipFile(images_zip_path) as zf:
        all_names = set(zf.namelist())
        for fname, text in wanted.items():
            # filenames in zip may or may not have a subdirectory prefix
            zip_key = fname
            if zip_key not in all_names:
                # try stripping path components
                base = Path(fname).name
                candidates = [n for n in all_names if Path(n).name == base]
                if candidates:
                    zip_key = candidates[0]
                else:
                    continue  # image missing from zip; skip

            dest = crops_dir / Path(zip_key).name
            data = zf.read(zip_key)
            dest.write_bytes(data)
            extracted += 1
            if extracted % 1000 == 0:
                print(f"  {extracted}/{len(wanted)} …", flush=True)

    print(f"  Extracted {extracted} images")

    if extracted == 0:
        sys.exit("No images extracted. Check that the zip is fully downloaded.")

    # --- Step 4: Build JSONL from extracted images ---
    rows: list[dict] = []
    for fname, text in wanted.items():
        dest = crops_dir / Path(fname).name
        if dest.exists() and text.strip():
            rows.append({"image": f"crops/{dest.name}", "text": text.strip()})

    random.seed(seed)
    random.shuffle(rows)

    val_count = max(1, int(len(rows) * val_ratio))
    train_rows = rows[val_count:]
    val_rows = rows[:val_count]

    def write_jsonl(path: Path, data: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)

    manifest = {
        "dataset_id": output_dir.name,
        "source": _DATASET_REPO,
        "subset": "lines_hu_v7",
        "created_at": _utc_now(),
        "total": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "splits": ["train", "val"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nDone. {len(train_rows)} train  +  {len(val_rows)} val  →  {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HuHTR synthetic Hungarian data")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output dataset directory (will be created)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20_000,
        help="Total number of samples to use (default: 20000, max ~100k)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction held out as validation (default: 0.05)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.count < 10:
        sys.exit("--count must be at least 10")
    if not (0.01 <= args.val_ratio <= 0.5):
        sys.exit("--val-ratio must be between 0.01 and 0.5")

    download(
        output_dir=args.output.expanduser().resolve(),
        total_count=args.count,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
