#!/usr/bin/env python3
"""
Export ground truth dataset to the shared folder for GPU training.

Produces a versioned dataset directory:
  SHARED/datasets/{dataset_id}/
    manifest.json       ← metadata and counts
    train.jsonl         ← {"line_id": N, "image": "crops/N.png", "text": "..."}
    val.jsonl
    test.jsonl
    crops/              ← copied image files (so GPU can rsync once)

Requires:
  - splits already assigned via assign_splits.py
  - INKWELL_SHARED env var or --shared argument pointing to the shared folder

Idempotent by default: if dataset_id folder already exists, aborts unless --force.

Usage:
    export INKWELL_SHARED=/mnt/share/inkwell
    python scripts/export_gt.py
    python scripts/export_gt.py --dataset-id gt_v2 --shared /mnt/share/inkwell
    python scripts/export_gt.py --force   # overwrite existing export
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inkwell.db import get_connection, DEFAULT_DB_PATH


def get_shared_path(override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("INKWELL_SHARED")
    if env:
        return Path(env).expanduser().resolve()
    # Fallback: local working/shared (useful for dev/testing)
    fallback = DEFAULT_DB_PATH.parent / "shared"
    print(f"[warn] INKWELL_SHARED not set, using fallback: {fallback}")
    return fallback


def get_gt_lines(conn) -> list[dict]:
    """All HUMAN_CORRECTED lines that have a split assignment."""
    rows = conn.execute("""
        SELECT
            t.line_id,
            t.text,
            l.crop_image_path,
            l.page_id,
            ds.split
        FROM transcriptions t
        JOIN lines l ON l.id = t.line_id
        JOIN dataset_splits ds ON ds.page_id = l.page_id
        WHERE t.transcription_type = 'HUMAN_CORRECTED'
          AND t.immutable = 1
          AND t.text NOT IN ('[ur]', '[nt]', '[?]')
          AND (t.flag IS NULL OR t.flag NOT IN ('UNUSABLE_SEGMENTATION', 'NOT_TEXT'))
          AND l.skip = 0
          AND l.crop_image_path IS NOT NULL
        ORDER BY ds.split, t.line_id
    """).fetchall()
    return [dict(r) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GT dataset to shared folder")
    parser.add_argument("--shared", default=None, help="Shared folder path (overrides INKWELL_SHARED)")
    parser.add_argument("--dataset-id", default=None, help="Dataset version ID (default: gt_YYYYMMDD)")
    parser.add_argument("--db", default=None, help="Override DB path")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset export")
    args = parser.parse_args()

    db_path = args.db or str(DEFAULT_DB_PATH)
    shared = get_shared_path(args.shared)
    dataset_id = args.dataset_id or f"gt_{datetime.now().strftime('%Y%m%d')}"

    conn = get_connection(db_path)
    working_dir = Path(db_path).parent
    line_crops_dir = working_dir / "line_crops"

    rows = get_gt_lines(conn)
    if not rows:
        print("No GT lines with split assignments found.")
        print("Run assign_splits.py first.")
        sys.exit(1)

    dataset_dir = shared / "datasets" / dataset_id
    if dataset_dir.exists() and not args.force:
        print(f"Dataset already exists: {dataset_dir}")
        print("Use --force to overwrite, or choose a different --dataset-id.")
        sys.exit(1)

    crops_out = dataset_dir / "crops"
    crops_out.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    missing = 0

    for row in rows:
        crop_name = row["crop_image_path"]
        src = line_crops_dir / crop_name
        if not src.exists():
            missing += 1
            continue
        dest_name = f"{row['line_id']}.png"
        shutil.copy2(src, crops_out / dest_name)
        entry = {
            "line_id": row["line_id"],
            "image": f"crops/{dest_name}",
            "text": row["text"],
            "page_id": row["page_id"],
        }
        splits[row["split"]].append(entry)

    counts: dict[str, int] = {}
    for split, entries in splits.items():
        out_file = dataset_dir / f"{split}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        counts[split] = len(entries)

    manifest = {
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "total": sum(counts.values()),
        "missing_crops": missing,
        "source_db": db_path,
    }
    with open(dataset_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDataset exported: {dataset_dir}")
    print(f"  train: {counts['train']}")
    print(f"  val:   {counts['val']}")
    print(f"  test:  {counts['test']}")
    print(f"  total: {sum(counts.values())}")
    if missing:
        print(f"  [warn] missing crop images: {missing}")
    print(f"\nTo rsync to GPU machine:")
    print(f"  rsync -avz --progress {shared}/ GPU_USER@GPU_HOST:{shared}/")


if __name__ == "__main__":
    main()
