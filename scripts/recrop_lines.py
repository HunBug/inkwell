#!/usr/bin/env python3
"""
Re-crop existing line images using current line polygons, with extra margins.

Safe-by-default behavior:
- preserves line IDs and DB matching
- does NOT change segmentation/line records
- overwrites only crop image files
- defaults to unannotated lines only

Typical use:
    python scripts/recrop_lines.py --top-extra 10 --only-unannotated
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inkwell.db import get_connection, DEFAULT_DB_PATH

JPEG_QUALITY = 95


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-crop line images with updated margins")
    p.add_argument("--db", default=None, help="Override DB path")
    p.add_argument("--top-extra", type=int, default=8, help="Extra pixels added above existing crop bbox")
    p.add_argument("--bottom-extra", type=int, default=0, help="Extra pixels added below existing crop bbox")
    p.add_argument("--left-extra", type=int, default=0, help="Extra pixels added left")
    p.add_argument("--right-extra", type=int, default=0, help="Extra pixels added right")
    p.add_argument("--only-unannotated", action="store_true", default=True, help="Only recrop lines without HUMAN_CORRECTED")
    p.add_argument("--include-annotated", action="store_true", help="Also recrop already annotated lines")
    p.add_argument("--limit", type=int, default=None, help="Limit number of lines to process")
    p.add_argument("--dry-run", action="store_true", help="Show counts only; do not write files")
    return p.parse_args()


def load_candidates(conn, include_annotated: bool, limit: int | None) -> list[dict]:
    where_unannotated = "" if include_annotated else """
      AND NOT EXISTS (
          SELECT 1
          FROM transcriptions hc
          WHERE hc.line_id = l.id
            AND hc.transcription_type = 'HUMAN_CORRECTED'
            AND hc.immutable = 1
      )
    """
    limit_sql = f"LIMIT {int(limit)}" if limit else ""

    sql = f"""
    SELECT
        l.id                AS line_id,
        l.crop_image_path   AS crop_image_path,
        l.polygon_coords    AS polygon_coords,
        p.derived_image_path AS page_image_path
    FROM lines l
    JOIN pages p ON p.id = l.page_id
    WHERE l.skip = 0
      AND l.crop_image_path IS NOT NULL
      AND l.polygon_coords IS NOT NULL
      AND p.derived_image_path IS NOT NULL
      {where_unannotated}
    ORDER BY l.id
    {limit_sql}
    """
    rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).expanduser().resolve() if args.db else DEFAULT_DB_PATH
    working_dir = db_path.parent
    line_crops_dir = working_dir / "line_crops"

    conn = get_connection(db_path)
    include_annotated = bool(args.include_annotated)
    candidates = load_candidates(conn, include_annotated, args.limit)
    conn.close()

    if not candidates:
        print("No lines matched recrop criteria.")
        return

    print(f"Candidates: {len(candidates)}")
    print(
        f"Margins: top+{args.top_extra}, bottom+{args.bottom_extra}, "
        f"left+{args.left_extra}, right+{args.right_extra}"
    )

    if args.dry_run:
        print("Dry-run only. No files written.")
        return

    updated = 0
    missing_page = 0
    missing_crop = 0
    bad_polygon = 0

    for i, row in enumerate(candidates, 1):
        page_path = working_dir / row["page_image_path"]
        crop_path = line_crops_dir / row["crop_image_path"]

        if not page_path.exists():
            missing_page += 1
            continue
        if not crop_path.exists():
            missing_crop += 1
            continue

        try:
            poly = json.loads(row["polygon_coords"])
            xs = [int(pt[0]) for pt in poly]
            ys = [int(pt[1]) for pt in poly]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        except Exception:
            bad_polygon += 1
            continue

        image = cv2.imread(str(page_path))
        if image is None:
            missing_page += 1
            continue

        h, w = image.shape[:2]
        cx1 = max(0, x1 - args.left_extra)
        cy1 = max(0, y1 - args.top_extra)
        cx2 = min(w - 1, x2 + args.right_extra)
        cy2 = min(h - 1, y2 + args.bottom_extra)

        if cx2 <= cx1 or cy2 <= cy1:
            bad_polygon += 1
            continue

        crop = image[cy1:cy2 + 1, cx1:cx2 + 1]
        ok = cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            updated += 1

        if i % 2000 == 0:
            print(f"  processed {i}/{len(candidates)}  updated={updated}")

    print("\nRecrop complete")
    print(f"  updated: {updated}")
    print(f"  missing page images: {missing_page}")
    print(f"  missing crop files: {missing_crop}")
    print(f"  bad polygons: {bad_polygon}")


if __name__ == "__main__":
    main()
