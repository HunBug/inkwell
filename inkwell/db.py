from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "working" / "inkwell.db"


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS config (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notebooks (
  id           INTEGER PRIMARY KEY,
  label        TEXT,
  folder_name  TEXT NOT NULL UNIQUE,
  date_start   TEXT,
  date_end     TEXT,
  notes        TEXT
);

CREATE TABLE IF NOT EXISTS assets (
  id          INTEGER PRIMARY KEY,
  notebook_id INTEGER NOT NULL REFERENCES notebooks(id),
  filename    TEXT NOT NULL,
  file_order  INTEGER NOT NULL,
  asset_type  TEXT NOT NULL DEFAULT 'DIARY_PAGE',
  should_ocr  INTEGER NOT NULL DEFAULT 1,
  notes       TEXT,
  UNIQUE(notebook_id, filename)
);

CREATE TABLE IF NOT EXISTS source_images (
  id                    INTEGER PRIMARY KEY,
  asset_id              INTEGER NOT NULL REFERENCES assets(id),
  orientation_detected  INTEGER,
  orientation_confirmed INTEGER,
  layout_type           TEXT,
  derived_image_path    TEXT,
  notes                 TEXT,
  UNIQUE(asset_id)
);

CREATE TABLE IF NOT EXISTS pages (
  id                  INTEGER PRIMARY KEY,
  source_image_id     INTEGER NOT NULL REFERENCES source_images(id),
  side                TEXT NOT NULL DEFAULT 'FULL',
  page_type           TEXT NOT NULL DEFAULT 'text',
  processing_status   TEXT NOT NULL DEFAULT 'pending',
  force_reprocess     INTEGER NOT NULL DEFAULT 0,
  derived_image_path  TEXT,
  notes               TEXT,
  UNIQUE(source_image_id, side)
);

CREATE TABLE IF NOT EXISTS segmentations (
  id                INTEGER PRIMARY KEY,
  page_id           INTEGER NOT NULL REFERENCES pages(id),
  segmentation_type TEXT NOT NULL,
  line_polygons     TEXT,
  model_version     TEXT,
  created_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  immutable         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS lines (
  id                      INTEGER PRIMARY KEY,
  page_id                 INTEGER NOT NULL REFERENCES pages(id),
  segmentation_id         INTEGER NOT NULL REFERENCES segmentations(id),
  line_order              INTEGER NOT NULL,
  polygon_coords          TEXT,
  crop_image_path         TEXT,
  segmentation_confidence REAL,
  skip                    INTEGER NOT NULL DEFAULT 0,
  skip_reason             TEXT,
  UNIQUE(segmentation_id, line_order)
);

CREATE TABLE IF NOT EXISTS transcriptions (
  id                 INTEGER PRIMARY KEY,
  line_id            INTEGER NOT NULL REFERENCES lines(id),
  transcription_type TEXT NOT NULL,
  text               TEXT NOT NULL,
  confidence         REAL,
  model_version      TEXT,
  created_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  created_by         TEXT NOT NULL,
  immutable          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS dataset_splits (
  id          INTEGER PRIMARY KEY,
  page_id     INTEGER NOT NULL REFERENCES pages(id),
  split       TEXT NOT NULL,
  assigned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(page_id)
);

CREATE TRIGGER IF NOT EXISTS protect_immutable_transcriptions_update
BEFORE UPDATE ON transcriptions
WHEN OLD.immutable = 1
BEGIN
  SELECT RAISE(ABORT, 'Cannot modify immutable transcription');
END;

CREATE TRIGGER IF NOT EXISTS protect_immutable_transcriptions_delete
BEFORE DELETE ON transcriptions
WHEN OLD.immutable = 1
BEGIN
  SELECT RAISE(ABORT, 'Cannot delete immutable transcription');
END;

CREATE TRIGGER IF NOT EXISTS protect_immutable_segmentations_update
BEFORE UPDATE ON segmentations
WHEN OLD.immutable = 1
BEGIN
  SELECT RAISE(ABORT, 'Cannot modify immutable segmentation');
END;

CREATE TRIGGER IF NOT EXISTS protect_immutable_segmentations_delete
BEFORE DELETE ON segmentations
WHEN OLD.immutable = 1
BEGIN
  SELECT RAISE(ABORT, 'Cannot delete immutable segmentation');
END;
"""


def get_connection(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()
