from __future__ import annotations

import sqlite3
from pathlib import Path


ROOT_KEY = "root_path"


def set_config(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO config(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    conn.commit()


def get_config(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_root_path(conn: sqlite3.Connection, path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    set_config(conn, ROOT_KEY, str(resolved))
    return resolved


def get_root_path(conn: sqlite3.Connection, override: str | Path | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()

    configured = get_config(conn, ROOT_KEY)
    if not configured:
        raise RuntimeError(
            "Root path is not configured. Run scripts/init_db.py or scripts/set_root.py first."
        )
    return Path(configured).expanduser().resolve()


def to_relative_path(path: str | Path, root_path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(Path(root_path).resolve()))


def resolve_path(relative_path: str | Path, root_path: str | Path) -> Path:
    return Path(root_path).expanduser().resolve() / Path(relative_path)
