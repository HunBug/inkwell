from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_polygon_coords(value: str | list[Any]) -> list[tuple[int, int]]:
    """Parse polygon coordinates from JSON string or list form."""
    raw = json.loads(value) if isinstance(value, str) else value
    points: list[tuple[int, int]] = []
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError("Invalid polygon point")
        points.append((int(point[0]), int(point[1])))
    if not points:
        raise ValueError("Empty polygon")
    return points


def polygon_bounds(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Return inclusive bbox (x1, y1, x2, y2) from polygon points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def expanded_clamped_bounds(
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
    top_extra: int = 0,
    bottom_extra: int = 0,
    left_extra: int = 0,
    right_extra: int = 0,
) -> tuple[int, int, int, int] | None:
    """Expand bbox by margins and clamp to image bounds (inclusive)."""
    cx1 = max(0, x1 - left_extra)
    cy1 = max(0, y1 - top_extra)
    cx2 = min(img_w - 1, x2 + right_extra)
    cy2 = min(img_h - 1, y2 + bottom_extra)

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    return cx1, cy1, cx2, cy2


def bounds_from_polygon(
    points: list[tuple[int, int]],
    *,
    img_w: int,
    img_h: int,
    top_extra: int = 0,
    bottom_extra: int = 0,
    left_extra: int = 0,
    right_extra: int = 0,
) -> tuple[int, int, int, int] | None:
    """Compute inclusive clamped bounds from polygon + margins."""
    x1, y1, x2, y2 = polygon_bounds(points)
    return expanded_clamped_bounds(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        img_w=img_w,
        img_h=img_h,
        top_extra=top_extra,
        bottom_extra=bottom_extra,
        left_extra=left_extra,
        right_extra=right_extra,
    )


def resolve_line_crop_path(
    raw_path: str,
    *,
    working_dir: Path,
    project_root: Path,
) -> Path:
    """Resolve line crop path from DB to an existing file path when possible."""
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p

    candidates = [
        working_dir / p,
        project_root / p,
        working_dir / "line_crops" / p.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def load_segmentation_tuning_config(conn, config_id: int | None = None) -> dict | None:
    """Load one saved segmentation tuning config row with parsed params.

    Returns None if no matching row is found.
    """
    if config_id is None:
        row = conn.execute(
            """
            SELECT id, segmenter, params_json, issue_n, clean_n, notes, created_at
            FROM segmentation_tuning_configs
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT id, segmenter, params_json, issue_n, clean_n, notes, created_at
            FROM segmentation_tuning_configs
            WHERE id = ?
            LIMIT 1
            """,
            (int(config_id),),
        ).fetchone()

    if not row:
        return None

    try:
        params = json.loads(row["params_json"])
    except Exception:
        params = {}

    def _as_int(name: str, default: int) -> int:
        try:
            return int(params.get(name, default))
        except Exception:
            return default

    normalized_params = {
        "top_extra": _as_int("top_extra", 0),
        "bottom_extra": _as_int("bottom_extra", 0),
        "left_extra": _as_int("left_extra", 0),
        "right_extra": _as_int("right_extra", 0),
    }

    return {
        "id": int(row["id"]),
        "segmenter": row["segmenter"],
        "params": normalized_params,
        "issue_n": int(row["issue_n"] or 0),
        "clean_n": int(row["clean_n"] or 0),
        "notes": row["notes"],
        "created_at": row["created_at"],
    }
