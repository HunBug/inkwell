"""
Phase 4: OCR

Run OCR on segmented line crops and store transcriptions in the database.

Current backend:
  - easyocr (CPU-friendly, NN-based, supports Hungarian)

Design note:
  The backend is selected by `model` and can be swapped later.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from inkwell.db import get_connection

logger = logging.getLogger(__name__)

OCR_TRANSCRIPTION_TYPE = "OCR_AUTO"


def _build_easyocr_reader(languages: list[str]):
    import easyocr

    return easyocr.Reader(languages, gpu=False)


def _run_easyocr(reader: Any, image_path: Path) -> tuple[str, float | None]:
    results = reader.readtext(str(image_path), detail=1, paragraph=False)
    if not results:
        return "", None

    parts: list[str] = []
    confidences: list[float] = []
    for item in results:
        if len(item) < 3:
            continue
        text = str(item[1]).strip()
        conf = item[2]
        if text:
            parts.append(text)
        if isinstance(conf, (float, int)):
            confidences.append(float(conf))

    merged_text = " ".join(parts).strip()
    avg_conf = (sum(confidences) / len(confidences)) if confidences else None
    return merged_text, avg_conf


def _delete_existing_ocr(conn, line_ids: list[int], transcription_type: str) -> None:
    if not line_ids:
        return

    placeholders = ",".join(["?"] * len(line_ids))
    conn.execute(
        f"""
        DELETE FROM transcriptions
        WHERE transcription_type = ?
          AND line_id IN ({placeholders})
        """,
        [transcription_type, *line_ids],
    )
    conn.commit()


def _get_lines_to_ocr(conn, page: int | None, limit: int | None, force: bool):
    base_where = ["l.skip = 0"]
    params: list[Any] = []

    if page is not None:
        base_where.append("l.page_id = ?")
        params.append(page)

    where_clause = " AND ".join(base_where)

    if force:
        query = f"""
            SELECT l.id, l.page_id, l.line_order, l.crop_image_path
            FROM lines l
            WHERE {where_clause}
            ORDER BY l.page_id, l.line_order
        """
    else:
        query = f"""
            SELECT l.id, l.page_id, l.line_order, l.crop_image_path
            FROM lines l
            WHERE {where_clause}
              AND NOT EXISTS (
                SELECT 1
                FROM transcriptions t
                WHERE t.line_id = l.id
                  AND t.transcription_type = ?
              )
            ORDER BY l.page_id, l.line_order
        """
        params.append(OCR_TRANSCRIPTION_TYPE)

    if limit is not None and limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    return conn.execute(query, params).fetchall()


def run_ocr(
    db_path: str,
    model: str = "easyocr",
    languages: list[str] | None = None,
    page: int | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, int]:
    """
    Run OCR for line crops.

    Args:
        db_path: Path to SQLite DB.
        model: OCR backend name (currently: easyocr).
        languages: OCR language list (EasyOCR language codes), defaults to ['hu', 'en'].
        page: Optional page_id filter.
        limit: Optional max number of lines to process.
        force: If True, remove existing OCR_AUTO transcriptions for selected lines first.
    """
    conn = get_connection(db_path)
    working_dir = Path(db_path).parent
    line_crops_dir = working_dir / "line_crops"

    if languages is None:
        languages = ["hu", "en"]

    selected_rows = _get_lines_to_ocr(conn, page=page, limit=limit, force=force)
    if not selected_rows:
        return {
            "processed": 0,
            "errors": 0,
            "empty": 0,
        }

    if force:
        line_ids = [int(row["id"]) for row in selected_rows]
        _delete_existing_ocr(conn, line_ids, OCR_TRANSCRIPTION_TYPE)

    if model != "easyocr":
        raise ValueError(f"Unsupported OCR model '{model}'. Supported: easyocr")

    reader = _build_easyocr_reader(languages)
    model_version = f"easyocr:{'+'.join(languages)}"

    stats = {
        "processed": 0,
        "errors": 0,
        "empty": 0,
    }

    try:
        for row in tqdm(selected_rows, desc="Running OCR", unit="line"):
            line_id = int(row["id"])
            crop_name = row["crop_image_path"]

            if not crop_name:
                stats["errors"] += 1
                logger.error("Line %s has no crop_image_path", line_id)
                continue

            image_path = line_crops_dir / crop_name
            if not image_path.exists():
                stats["errors"] += 1
                logger.error("Missing crop image for line %s: %s", line_id, image_path)
                continue

            try:
                text, confidence = _run_easyocr(reader, image_path)
                if not text:
                    stats["empty"] += 1

                conn.execute(
                    """
                    INSERT INTO transcriptions (
                        line_id,
                        transcription_type,
                        text,
                        confidence,
                        model_version,
                        created_by,
                        immutable
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        line_id,
                        OCR_TRANSCRIPTION_TYPE,
                        text,
                        confidence,
                        model_version,
                        model,
                    ),
                )
                conn.commit()
                stats["processed"] += 1
            except Exception as exc:
                conn.rollback()
                stats["errors"] += 1
                logger.error("OCR failed for line %s: %s", line_id, exc)
    except KeyboardInterrupt:
        conn.commit()
        logger.warning("OCR interrupted by user. Partial progress is saved.")

    return stats