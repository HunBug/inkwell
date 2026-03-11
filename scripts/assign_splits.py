#!/usr/bin/env python3
"""
Assign page-level dataset splits (train/val/test) incrementally.

Only pages that have at least one HUMAN_CORRECTED transcription are eligible.
Pages already in dataset_splits are skipped — splits are never reshuffled.
New pages are assigned to maintain a ~70/15/15 ratio given what already exists.

Usage:
    python scripts/assign_splits.py [--dry-run]
    python scripts/assign_splits.py --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inkwell.db import get_connection, DEFAULT_DB_PATH


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15  (remainder)


def get_eligible_unassigned_pages(conn) -> list[int]:
    """Pages with ≥1 HUMAN_CORRECTED line that don't have a split yet."""
    rows = conn.execute("""
        SELECT DISTINCT p.id
        FROM pages p
        JOIN lines l ON l.page_id = p.id
        JOIN transcriptions t ON t.line_id = l.id
        WHERE t.transcription_type = 'HUMAN_CORRECTED'
          AND NOT EXISTS (
              SELECT 1 FROM dataset_splits ds WHERE ds.page_id = p.id
          )
        ORDER BY p.id
    """).fetchall()
    return [r["id"] for r in rows]


def get_current_split_counts(conn) -> dict[str, int]:
    rows = conn.execute("""
        SELECT split, COUNT(*) as cnt FROM dataset_splits GROUP BY split
    """).fetchall()
    counts = {"train": 0, "val": 0, "test": 0}
    for r in rows:
        counts[r["split"]] = r["cnt"]
    return counts


def assign_split_for_new_pages(
    page_ids: list[int],
    current_counts: dict[str, int],
    seed: int,
) -> list[tuple[int, str]]:
    """
    Deterministically assign splits to new pages, respecting current ratio.
    Uses page_id order (already sorted) + seed offset to spread across buckets.
    """
    import random
    rng = random.Random(seed)
    shuffled = list(page_ids)
    rng.shuffle(shuffled)

    assignments: list[tuple[int, str]] = []
    counts = dict(current_counts)

    for page_id in shuffled:
        total = sum(counts.values()) + 1  # +1 for this page
        train_target = int(total * TRAIN_RATIO)
        val_target = int(total * VAL_RATIO)

        if counts["train"] < train_target:
            split = "train"
        elif counts["val"] < val_target:
            split = "val"
        else:
            split = "test"

        assignments.append((page_id, split))
        counts[split] += 1

    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign incremental page-level dataset splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="Print assignments without writing")
    parser.add_argument("--db", default=None, help="Override DB path")
    args = parser.parse_args()

    db_path = args.db or str(DEFAULT_DB_PATH)
    conn = get_connection(db_path)

    current_counts = get_current_split_counts(conn)
    new_pages = get_eligible_unassigned_pages(conn)

    print(f"Already assigned: train={current_counts['train']}  val={current_counts['val']}  test={current_counts['test']}")
    print(f"New eligible pages: {len(new_pages)}")

    if not new_pages:
        print("Nothing to assign. Annotate more lines first.")
        return

    assignments = assign_split_for_new_pages(new_pages, current_counts, args.seed)

    split_tally = {"train": 0, "val": 0, "test": 0}
    for page_id, split in assignments:
        split_tally[split] += 1
        if args.dry_run:
            print(f"  page {page_id:5d} → {split}")

    print(f"\nWill assign: train={split_tally['train']}  val={split_tally['val']}  test={split_tally['test']}")

    if args.dry_run:
        print("\n[dry-run] No changes written.")
        return

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT OR IGNORE INTO dataset_splits (page_id, split, assigned_at) VALUES (?, ?, ?)",
        [(page_id, split, now) for page_id, split in assignments],
    )
    conn.commit()

    final = get_current_split_counts(conn)
    print(f"\nDone. Total splits: train={final['train']}  val={final['val']}  test={final['test']}")


if __name__ == "__main__":
    main()
