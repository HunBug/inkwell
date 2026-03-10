"""
Phase 4: OCR

Run OCR on segmented line crops and store transcriptions in the database.

Current backends:
    - easyocr (CPU-friendly, NN-based, supports Hungarian)
    - trocr (transformer OCR for handwriting)

Design note:
  The backend is selected by `model` and can be swapped later.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

from tqdm import tqdm

from inkwell.db import get_connection

logger = logging.getLogger(__name__)

OCR_TRANSCRIPTION_TYPE = "OCR_AUTO"
DEFAULT_TROCR_CHECKPOINT = "microsoft/trocr-base-handwritten"


def _build_easyocr_reader(languages: list[str]):
    import easyocr

    return easyocr.Reader(languages, gpu=False)


def _build_trocr(checkpoint: str = DEFAULT_TROCR_CHECKPOINT):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    processor = TrOCRProcessor.from_pretrained(checkpoint, use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
    return processor, model


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


def _run_trocr(
    processor: Any,
    model: Any,
    image_path: Path,
) -> tuple[str, float | None]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return text, None


def _delete_existing_ocr(
    conn,
    line_ids: list[int],
    transcription_type: str,
    model_key: str,
) -> None:
    if not line_ids:
        return

    placeholders = ",".join(["?"] * len(line_ids))
    conn.execute(
        f"""
        DELETE FROM transcriptions
        WHERE transcription_type = ?
                    AND created_by = ?
          AND line_id IN ({placeholders})
        """,
                [transcription_type, model_key, *line_ids],
    )
    conn.commit()


def _get_lines_to_ocr(
    conn,
    page: int | None,
    limit: int | None,
    force: bool,
    model_key: str,
):
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
                  AND t.created_by = ?
              )
            ORDER BY l.page_id, l.line_order
        """
        params.append(OCR_TRANSCRIPTION_TYPE)
        params.append(model_key)

    if limit is not None and limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    return conn.execute(query, params).fetchall()


def _resolve_model(model: str) -> tuple[str, str, str]:
    raw = model.strip()
    normalized = raw.lower()

    if normalized == "easyocr":
        return "easyocr", "easyocr", "easyocr"

    if normalized in {"trocr", "trocr-base"}:
        checkpoint = DEFAULT_TROCR_CHECKPOINT
        return "trocr", checkpoint, f"trocr:{checkpoint}"

    if normalized == "trocr-large":
        checkpoint = "microsoft/trocr-large-handwritten"
        return "trocr", checkpoint, f"trocr:{checkpoint}"

    if raw.startswith("trocr:") or raw.startswith("trocr@"):
        checkpoint = raw.split(":", 1)[1] if raw.startswith("trocr:") else raw.split("@", 1)[1]
        if not checkpoint:
            raise ValueError("Custom TrOCR model must include checkpoint after 'trocr:' or 'trocr@'")
        return "trocr", checkpoint, f"trocr:{checkpoint}"

    raise ValueError(
        "Unsupported OCR model "
        f"'{model}'. Supported: easyocr, trocr, trocr-base, trocr-large, trocr:<checkpoint>"
    )


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
        model: OCR backend name (easyocr, trocr, trocr-base, trocr-large).
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

    warnings.filterwarnings(
        "ignore",
        message=r".*pin_memory.*no accelerator is found.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"torch\.utils\.data\.dataloader",
    )

    backend, backend_config, model_version = _resolve_model(model)
    model_key = model_version

    selected_rows = _get_lines_to_ocr(
        conn,
        page=page,
        limit=limit,
        force=force,
        model_key=model_key,
    )
    if not selected_rows:
        return {
            "processed": 0,
            "errors": 0,
            "empty": 0,
        }

    if force:
        line_ids = [int(row["id"]) for row in selected_rows]
        _delete_existing_ocr(conn, line_ids, OCR_TRANSCRIPTION_TYPE, model_key)

    easyocr_reader = None
    trocr_processor = None
    trocr_model = None
    if backend == "easyocr":
        easyocr_reader = _build_easyocr_reader(languages)
        model_version = f"easyocr:{'+'.join(languages)}"
    elif backend == "trocr":
        trocr_processor, trocr_model = _build_trocr(backend_config)

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
                if backend == "easyocr":
                    text, confidence = _run_easyocr(easyocr_reader, image_path)
                else:
                    text, confidence = _run_trocr(trocr_processor, trocr_model, image_path)
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
                        model_key,
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