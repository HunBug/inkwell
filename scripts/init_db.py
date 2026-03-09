#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inkwell.config import set_root_path
from inkwell.db import create_schema, get_connection


COUNTER_PATTERN = re.compile(r"(?:_|-)(\d{3,6})(?=\.[^.]+$)")


def extract_file_order(filename: str) -> int:
    match = COUNTER_PATTERN.search(filename)
    if match:
        return int(match.group(1))
    return 999_999


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize inkwell database and ingest assets")
    parser.add_argument("--config", required=True, help="Path to notebooks_config.json")
    parser.add_argument("--db", default=None, help="Override DB path (default: working/inkwell.db)")
    parser.add_argument("--root", default=None, help="Override root_path from config file")
    return parser.parse_args()


def upsert_notebook(conn, notebook: dict) -> int:
    conn.execute(
        """
        INSERT INTO notebooks(label, folder_name, date_start, date_end, notes)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(folder_name) DO UPDATE SET
            label = excluded.label,
            date_start = excluded.date_start,
            date_end = excluded.date_end,
            notes = excluded.notes
        """,
        (
            notebook.get("label") or notebook["folder_name"],
            notebook["folder_name"],
            notebook.get("date_start"),
            notebook.get("date_end"),
            notebook.get("notes"),
        ),
    )
    row = conn.execute(
        "SELECT id FROM notebooks WHERE folder_name = ?", (notebook["folder_name"],)
    ).fetchone()
    return row["id"]


def ingest_assets_for_notebook(conn, root_path: Path, notebook_id: int, folder_name: str) -> tuple[int, int]:
    folder_path = root_path / folder_name
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Notebook folder does not exist: {folder_path}")

    files = [
        entry
        for entry in folder_path.iterdir()
        if entry.is_file() and entry.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    ]
    files.sort(key=lambda p: (extract_file_order(p.name), p.name.lower()))

    inserted_assets = 0
    inserted_source_images = 0

    for file_path in files:
        file_order = extract_file_order(file_path.name)
        conn.execute(
            """
            INSERT OR IGNORE INTO assets(notebook_id, filename, file_order, asset_type, should_ocr, notes)
            VALUES (?, ?, ?, 'DIARY_PAGE', 1, NULL)
            """,
            (notebook_id, file_path.name, file_order),
        )

        asset_row = conn.execute(
            "SELECT id FROM assets WHERE notebook_id = ? AND filename = ?",
            (notebook_id, file_path.name),
        ).fetchone()

        inserted_assets += conn.execute("SELECT changes()").fetchone()[0]

        conn.execute(
            """
            INSERT OR IGNORE INTO source_images(asset_id, orientation_detected, orientation_confirmed, layout_type, derived_image_path, notes)
            VALUES (?, NULL, NULL, NULL, NULL, NULL)
            """,
            (asset_row["id"],),
        )
        inserted_source_images += conn.execute("SELECT changes()").fetchone()[0]

        source_image_row = conn.execute(
            "SELECT id FROM source_images WHERE asset_id = ?", (asset_row["id"],)
        ).fetchone()

        conn.execute(
            """
            INSERT OR IGNORE INTO pages(source_image_id, side, page_type, processing_status, force_reprocess, notes)
            VALUES (?, 'FULL', 'text', 'pending', 0, NULL)
            """,
            (source_image_row["id"],),
        )

    return inserted_assets, inserted_source_images


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    notebooks = payload.get("notebooks") or []
    if not notebooks:
        raise RuntimeError("Config must include a non-empty 'notebooks' array")

    root_path_raw = args.root or payload.get("root_path")
    if not root_path_raw:
        raise RuntimeError("Root path missing. Provide --root or root_path in config.")

    conn = get_connection(args.db)
    create_schema(conn)

    root_path = set_root_path(conn, root_path_raw)

    total_assets_added = 0
    total_source_images_added = 0

    with conn:
        for notebook in notebooks:
            if "folder_name" not in notebook:
                raise RuntimeError("Each notebook entry must include folder_name")

            notebook_id = upsert_notebook(conn, notebook)
            assets_added, source_images_added = ingest_assets_for_notebook(
                conn, root_path, notebook_id, notebook["folder_name"]
            )
            total_assets_added += assets_added
            total_source_images_added += source_images_added

    notebook_count = conn.execute("SELECT COUNT(*) AS c FROM notebooks").fetchone()["c"]
    asset_count = conn.execute("SELECT COUNT(*) AS c FROM assets").fetchone()["c"]
    source_image_count = conn.execute("SELECT COUNT(*) AS c FROM source_images").fetchone()["c"]
    page_count = conn.execute("SELECT COUNT(*) AS c FROM pages").fetchone()["c"]

    print("Init complete")
    print(f"Root path: {root_path}")
    print(f"Notebooks: {notebook_count}")
    print(f"Assets: {asset_count} (added this run: {total_assets_added})")
    print(
        f"Source images: {source_image_count} (added this run: {total_source_images_added})"
    )
    print(f"Pages: {page_count}")


if __name__ == "__main__":
    main()
