#!/usr/bin/env python3
"""
Export full unlabeled line pool for GPU-side recognition.

Creates under shared datasets:
  SHARED/datasets/{dataset_id}/unlabeled_pool/
    manifest.json
    pool.jsonl
    crops/{line_id}.png

`pool.jsonl` rows:
  {
    "line_id": 123,
    "page_id": 45,
    "notebook_id": 6,
    "notebook_folder": "...",
    "ocr_text": "...",
    "ocr_confidence": 0.34,
    "image": "crops/123.png"
  }
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
from inkwell.cropping import load_segmentation_tuning_config, resolve_line_crop_path


def get_shared_path(override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("INKWELL_SHARED")
    if env:
        return Path(env).expanduser().resolve()
    fallback = DEFAULT_DB_PATH.parent / "shared"
    print(f"[warn] INKWELL_SHARED not set, using fallback: {fallback}")
    return fallback


def get_unlabeled_rows(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            l.id                      AS line_id,
            l.page_id                 AS page_id,
            nb.id                     AS notebook_id,
            nb.folder_name            AS notebook_folder,
            l.crop_image_path         AS crop_image_path,
            best.text                 AS ocr_text,
            best.confidence           AS ocr_confidence
        FROM lines l
        JOIN pages p            ON p.id = l.page_id
        JOIN source_images si   ON si.id = p.source_image_id
        JOIN assets a           ON a.id = si.asset_id
        JOIN notebooks nb       ON nb.id = a.notebook_id
        JOIN transcriptions best ON best.id = (
            SELECT t.id
            FROM transcriptions t
            WHERE t.line_id = l.id
              AND t.transcription_type = 'OCR_AUTO'
            ORDER BY t.confidence DESC NULLS LAST, t.id DESC
            LIMIT 1
        )
        WHERE l.skip = 0
          AND l.crop_image_path IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM transcriptions hc
              WHERE hc.line_id = l.id
                AND hc.transcription_type = 'HUMAN_CORRECTED'
                AND hc.immutable = 1
          )
          AND NOT EXISTS (
              SELECT 1
              FROM transcriptions fl
              WHERE fl.line_id = l.id
                AND fl.transcription_type = 'FLAGGED'
          )
        ORDER BY l.id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def main() -> None:
    p = argparse.ArgumentParser(description="Export full unlabeled pool for GPU inference")
    p.add_argument("--dataset-id", required=True, help="Dataset ID under shared/datasets/")
    p.add_argument("--shared", default=None, help="Shared folder path")
    p.add_argument("--db", default=None, help="Override DB path")
    p.add_argument("--force", action="store_true", help="Overwrite existing unlabeled_pool export")
    p.add_argument(
        "--tuning-config-id",
        type=int,
        default=None,
        help="Pin crop profile metadata to this segmentation_tuning_configs.id",
    )
    args = p.parse_args()

    shared = get_shared_path(args.shared)
    dataset_dir = shared / "datasets" / args.dataset_id
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    out_dir = dataset_dir / "unlabeled_pool"
    crops_out = out_dir / "crops"
    if out_dir.exists() and args.force:
            print(f"[info] --force: clearing existing export at {out_dir}", flush=True)
            # shutil.rmtree fails on network mounts with ENOTEMPTY; delete file-by-file instead
            if crops_out.exists():
                removed = 0
                for f in crops_out.iterdir():
                    if f.is_file():
                        f.unlink()
                        removed += 1
                crops_out.rmdir()
                print(f"[info] Removed {removed} existing crops", flush=True)
            for leftover in out_dir.iterdir():
                if leftover.is_file():
                    leftover.unlink()
            out_dir.rmdir()
    crops_out.mkdir(parents=True, exist_ok=True)

    conn = get_connection(args.db)
    tuning_config = load_segmentation_tuning_config(conn, args.tuning_config_id)
    rows = get_unlabeled_rows(conn)
    conn.close()

    working_dir = DEFAULT_DB_PATH.parent
    missing = 0
    exported = 0
    copied_new = 0
    total_rows = len(rows)

    with open(out_dir / "pool.jsonl", "w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, 1):
            src = resolve_line_crop_path(
                row["crop_image_path"],
                working_dir=working_dir,
                project_root=PROJECT_ROOT,
            )
            if not src.exists():
                missing += 1
                continue

            dst_name = f"{row['line_id']}{src.suffix or '.png'}"
            dst = crops_out / dst_name
            if not dst.exists():
                shutil.copyfile(src, dst)
                copied_new += 1

            item = {
                "line_id": row["line_id"],
                "page_id": row["page_id"],
                "notebook_id": row["notebook_id"],
                "notebook_folder": row["notebook_folder"],
                "ocr_text": row.get("ocr_text") or "",
                "ocr_confidence": row.get("ocr_confidence"),
                "image": f"crops/{dst_name}",
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            exported += 1

            if idx % 1000 == 0:
                print(
                    f"[export] {idx}/{total_rows} scanned | exported={exported} copied_new={copied_new} missing={missing}",
                    flush=True,
                )

    manifest = {
        "dataset_id": args.dataset_id,
        "kind": "unlabeled_pool",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_candidates": total_rows,
        "exported": exported,
        "copied_new": copied_new,
        "missing_crops": missing,
        "source_db": str(Path(args.db).expanduser().resolve()) if args.db else str(DEFAULT_DB_PATH),
        "pool_jsonl": str(out_dir / "pool.jsonl"),
        "crop_profile": tuning_config,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"Exported unlabeled pool: {out_dir}")
    print(f"  total candidates: {total_rows}")
    print(f"  lines: {exported}")
    print(f"  copied new crops: {copied_new}")
    print(f"  missing crops: {missing}")


if __name__ == "__main__":
    main()
