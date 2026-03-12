#!/usr/bin/env python3
"""One-time cleanup for common grave-accent keyboard mistakes in GT rows.

Rewrites HUMAN_CORRECTED transcription text:
    à -> á
    À -> Á
    è -> é
    È -> É
    ì -> í
    Ì -> Í

This script temporarily drops the immutable-update trigger, performs the text
normalization, then restores the trigger.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inkwell.db import DEFAULT_DB_PATH, get_connection


UPDATE_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS protect_immutable_transcriptions_update
BEFORE UPDATE ON transcriptions
WHEN OLD.immutable = 1
BEGIN
  SELECT RAISE(ABORT, 'Cannot modify immutable transcription');
END;
"""


def _count_candidates(conn) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM transcriptions
        WHERE transcription_type = 'HUMAN_CORRECTED'
          AND (
              instr(text, 'à') > 0 OR instr(text, 'À') > 0 OR
              instr(text, 'è') > 0 OR instr(text, 'È') > 0 OR
              instr(text, 'ì') > 0 OR instr(text, 'Ì') > 0
          )
        """
    ).fetchone()
    return int(row[0] or 0)


def normalize_gt_accents(conn) -> int:
    conn.execute("DROP TRIGGER IF EXISTS protect_immutable_transcriptions_update")
    try:
        before = conn.total_changes
        conn.execute(
            """
            UPDATE transcriptions
            SET text = REPLACE(
                REPLACE(
                    REPLACE(
                        REPLACE(
                            REPLACE(
                                REPLACE(text, 'à', 'á'),
                                'À', 'Á'
                            ),
                            'è', 'é'
                        ),
                        'È', 'É'
                    ),
                    'ì', 'í'
                ),
                'Ì', 'Í'
            )
            WHERE transcription_type = 'HUMAN_CORRECTED'
              AND (
                  instr(text, 'à') > 0 OR instr(text, 'À') > 0 OR
                  instr(text, 'è') > 0 OR instr(text, 'È') > 0 OR
                  instr(text, 'ì') > 0 OR instr(text, 'Ì') > 0
              )
            """
        )
        changed = conn.total_changes - before
        conn.commit()
        return changed
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.executescript(UPDATE_TRIGGER_SQL)
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize grave-accent keyboard mistakes in GT rows"
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help="SQLite DB path (default: working/inkwell.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report how many rows would be changed",
    )
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    conn = get_connection(db_path)
    try:
        candidates = _count_candidates(conn)
        print(f"DB: {db_path}")
        print(f"Rows needing normalization: {candidates}")
        if args.dry_run or candidates == 0:
            return

        changed = normalize_gt_accents(conn)
        print(f"Rows updated: {changed}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()