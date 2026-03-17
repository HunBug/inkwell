from __future__ import annotations

import io
import json
import os

from flask import Blueprint, render_template, request, jsonify, current_app, g, send_file
from pathlib import Path
from PIL import Image

from inkwell.db import get_connection, DEFAULT_DB_PATH
from inkwell.cropping import (
    bounds_from_polygon,
    parse_polygon_coords,
    polygon_bounds,
)
from inkwell.text_policy import (
    ensure_text_policy_table,
    load_text_policy_from_automation_toml,
    summarize_text_policy_rows,
)


annotate_bp = Blueprint("annotate", __name__, url_prefix="/annotate")


_POOL_PREDICTIONS_CACHE: dict = {
    "path": None,
    "mtime": None,
    "map": {},
}


ACCENT_NORMALIZATION = str.maketrans({
    "à": "á",
    "À": "Á",
    "è": "é",
    "È": "É",
    "ì": "í",
    "Ì": "Í",
})


def _normalize_annotation_text(text: str) -> str:
    """Fix common Estonian-keyboard grave/acute mixups before saving GT."""
    return text.translate(ACCENT_NORMALIZATION)


def get_db():
    """Get database connection for current request."""
    if "db" not in g:
        g.db = get_connection(current_app.config.get("DB_PATH"))
        # Ensure flag column exists (migration)
        _ensure_flag_column(g.db)
        _ensure_annotation_indexes(g.db)
        _ensure_segmentation_tuning_table(g.db)
        ensure_text_policy_table(g.db)
    return g.db


def _load_active_text_policy() -> dict:
    return load_text_policy_from_automation_toml(
        Path(current_app.root_path).parents[1] / "automation.toml"
    )


def _get_gt_rows_for_quality(db) -> list[dict]:
    rows = db.execute(
        """
                WITH latest_human AS (
                        SELECT line_id, MAX(id) AS latest_id
                        FROM transcriptions
                        WHERE transcription_type = 'HUMAN_CORRECTED'
                            AND immutable = 1
                        GROUP BY line_id
                )
        SELECT
            t.line_id,
            t.text,
            COALESCE(ds.split, 'train') AS split
                FROM latest_human lh
                JOIN transcriptions t ON t.id = lh.latest_id
        JOIN lines l ON l.id = t.line_id
        LEFT JOIN dataset_splits ds ON ds.page_id = l.page_id
                WHERE (t.flag IS NULL OR t.flag NOT IN ('UNUSABLE_SEGMENTATION', 'NOT_TEXT'))
          AND l.skip = 0
                ORDER BY t.id DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def _ensure_flag_column(db):
    """Add flag column to transcriptions table if it doesn't exist."""
    try:
        cursor = db.execute("PRAGMA table_info(transcriptions)")
        columns = {row[1] for row in cursor.fetchall()}
        if "flag" not in columns:
            db.execute("ALTER TABLE transcriptions ADD COLUMN flag TEXT")
            db.commit()
    except Exception:
        pass  # Column already exists or other error


def _ensure_annotation_indexes(db):
    """Add annotation-related indexes if they do not exist yet."""
    db.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_lines_skip_id
        ON lines(skip, id);

        CREATE INDEX IF NOT EXISTS idx_transcriptions_type_line
        ON transcriptions(transcription_type, line_id);

        CREATE INDEX IF NOT EXISTS idx_transcriptions_line_type_conf_id
        ON transcriptions(line_id, transcription_type, confidence DESC, id DESC);
        """
    )
    db.commit()


def _ensure_segmentation_tuning_table(db):
        """Create segmentation tuning config table if missing."""
        db.executescript(
                """
                CREATE TABLE IF NOT EXISTS segmentation_tuning_configs (
                    id                   INTEGER PRIMARY KEY,
                    source               TEXT NOT NULL DEFAULT 'segment_tuning_ui',
                    segmenter            TEXT NOT NULL DEFAULT 'cv_projection',
                    params_json          TEXT NOT NULL,
                    issue_n              INTEGER NOT NULL DEFAULT 0,
                    clean_n              INTEGER NOT NULL DEFAULT 0,
                    sample_line_ids_json TEXT,
                    notes                TEXT,
                    created_by           TEXT NOT NULL DEFAULT 'human',
                    created_at           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_seg_tuning_configs_created_at
                ON segmentation_tuning_configs(created_at DESC);
                """
        )
        db.commit()


def _get_random_unannotated_line(db):
    """Pick one random unannotated line that has OCR output."""
    return db.execute(
        """
        WITH candidate AS (
            SELECT DISTINCT t.line_id
            FROM transcriptions t
            JOIN lines l ON l.id = t.line_id
            WHERE t.transcription_type = 'OCR_AUTO'
              AND l.skip = 0
              AND NOT EXISTS (
                  SELECT 1
                  FROM transcriptions t2
                  WHERE t2.line_id = t.line_id
                    AND t2.transcription_type IN ('HUMAN_CORRECTED', 'FLAGGED')
              )
            ORDER BY RANDOM()
            LIMIT 1
        )
        SELECT
            l.id AS line_id,
            l.crop_image_path,
            best.text AS ocr_text,
            best.confidence AS confidence,
            best.model_version AS model_version
        FROM candidate c
        JOIN lines l ON l.id = c.line_id
        JOIN transcriptions best
          ON best.id = (
              SELECT t.id
              FROM transcriptions t
              WHERE t.line_id = c.line_id
                AND t.transcription_type = 'OCR_AUTO'
              ORDER BY t.confidence DESC, t.id DESC
              LIMIT 1
          )
        """
    ).fetchone()


def _working_root() -> Path:
    return DEFAULT_DB_PATH.parent


def _resolve_shared_root() -> Path | None:
    configured = current_app.config.get("INKWELL_SHARED") or os.environ.get("INKWELL_SHARED")
    if configured:
        p = Path(configured).expanduser().resolve()
        return p if p.exists() else None

    network_default = Path("/home/akoss/mnt/lara-playground/playground/inkwell-automation")
    if network_default.exists():
        return network_default

    local_fallback = _working_root() / "shared"
    return local_fallback if local_fallback.exists() else None


def _find_latest_infer_pool_predictions(shared: Path | None) -> Path | None:
    if shared is None or not shared.exists():
        return None

    jobs_dir = shared / "jobs"
    if not jobs_dir.exists():
        return None

    best: tuple[str, Path] | None = None
    for d in sorted(jobs_dir.iterdir()):
        if not d.name.startswith("infer_pool_"):
            continue

        result_file = d / "result.json"
        preds_file = d / "pool_predictions.jsonl"
        if not result_file.exists() or not preds_file.exists():
            continue

        try:
            result = json.loads(result_file.read_text())
        except Exception:
            continue

        if result.get("status") != "completed":
            continue

        finished_at = result.get("finished_at", "")
        if best is None or finished_at > best[0]:
            best = (finished_at, preds_file)

    return best[1] if best else None


def _load_latest_pool_predictions_map() -> dict[int, str]:
    shared = _resolve_shared_root()
    preds_path = _find_latest_infer_pool_predictions(shared)
    if preds_path is None:
        _POOL_PREDICTIONS_CACHE["path"] = None
        _POOL_PREDICTIONS_CACHE["mtime"] = None
        _POOL_PREDICTIONS_CACHE["map"] = {}
        return {}

    try:
        mtime = preds_path.stat().st_mtime
    except Exception:
        return {}

    if (
        _POOL_PREDICTIONS_CACHE.get("path") == str(preds_path)
        and _POOL_PREDICTIONS_CACHE.get("mtime") == mtime
    ):
        return _POOL_PREDICTIONS_CACHE.get("map", {})

    out: dict[int, str] = {}
    try:
        with open(preds_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    line_id = row.get("line_id")
                    pred = row.get("predicted_text")
                    if isinstance(line_id, int) and isinstance(pred, str):
                        out[line_id] = pred.strip()
                except Exception:
                    continue
    except Exception:
        out = {}

    _POOL_PREDICTIONS_CACHE["path"] = str(preds_path)
    _POOL_PREDICTIONS_CACHE["mtime"] = mtime
    _POOL_PREDICTIONS_CACHE["map"] = out
    return out


def _apply_live_pool_ocr(row):
    if not row:
        return row

    item = dict(row)
    predictions = _load_latest_pool_predictions_map()
    predicted = predictions.get(item.get("line_id"))

    if predicted:
        item["ocr_text"] = predicted
        item["confidence"] = None
        item["model_version"] = "infer_pool_latest"
        item["ocr_source"] = "pool_predictions"
    else:
        item["ocr_source"] = "db_ocr_auto"

    return item


def _suggestions_dir() -> Path:
    return _working_root() / "suggestions"


def _list_suggestion_files() -> list[str]:
    d = _suggestions_dir()
    if not d.exists():
        return []
    files = [p.name for p in d.glob("next_samples_*.jsonl") if p.is_file()]
    return sorted(files, reverse=True)


def _resolve_suggestion_file(filename: str | None) -> Path | None:
    if not filename:
        return None
    # only allow a plain filename under working/suggestions (no path traversal)
    safe_name = Path(filename).name
    if safe_name != filename:
        return None
    p = _suggestions_dir() / safe_name
    return p if p.exists() and p.is_file() else None


def _load_suggested_line_ids(file_path: Path) -> list[int]:
    ids: list[int] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                line_id = obj.get("line_id")
                if isinstance(line_id, int):
                    ids.append(line_id)
            except Exception:
                continue
    return ids


def _get_line_payload_by_id(db, line_id: int):
    return db.execute(
        """
        SELECT
            l.id AS line_id,
            l.crop_image_path,
            best.text AS ocr_text,
            best.confidence AS confidence,
            best.model_version AS model_version
        FROM lines l
        JOIN transcriptions best
          ON best.id = (
              SELECT t.id
              FROM transcriptions t
              WHERE t.line_id = l.id
                AND t.transcription_type = 'OCR_AUTO'
              ORDER BY t.confidence DESC, t.id DESC
              LIMIT 1
          )
        WHERE l.id = ?
          AND l.skip = 0
          AND NOT EXISTS (
              SELECT 1
              FROM transcriptions t2
              WHERE t2.line_id = l.id
                AND t2.transcription_type IN ('HUMAN_CORRECTED', 'FLAGGED')
          )
        LIMIT 1
        """,
        (line_id,),
    ).fetchone()


def _get_next_suggested_unannotated_line(db, suggestion_filename: str | None):
    suggestion_path = _resolve_suggestion_file(suggestion_filename)
    if not suggestion_path:
        return None

    line_ids = _load_suggested_line_ids(suggestion_path)
    if not line_ids:
        return None

    for line_id in line_ids:
        row = _get_line_payload_by_id(db, line_id)
        if row:
            return row
    return None


def _pick_line(db, suggestion_filename: str | None = None):
    if suggestion_filename:
        row = _get_next_suggested_unannotated_line(db, suggestion_filename)
    else:
        row = _get_random_unannotated_line(db)
    return _apply_live_pool_ocr(row)


def _parse_int_param(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value))


def _sample_segmentation_tuning_lines(db, issue_n: int, clean_n: int) -> list[dict]:
    issue_rows = db.execute(
        """
        WITH latest_human AS (
            SELECT t.line_id, t.flag, t.id
            FROM transcriptions t
            JOIN (
                SELECT line_id, MAX(id) AS latest_id
                FROM transcriptions
                WHERE transcription_type IN ('HUMAN_CORRECTED', 'FLAGGED')
                GROUP BY line_id
            ) latest ON latest.latest_id = t.id
        )
        SELECT
            l.id AS line_id,
            l.crop_image_path,
            p.derived_image_path,
            l.polygon_coords,
            COALESCE(latest_human.flag, '') AS flag
        FROM latest_human
        JOIN lines l ON l.id = latest_human.line_id
        JOIN pages p ON p.id = l.page_id
        WHERE l.skip = 0
          AND l.crop_image_path IS NOT NULL
          AND l.polygon_coords IS NOT NULL
          AND p.derived_image_path IS NOT NULL
          AND (
              latest_human.flag LIKE '%SEGMENTATION_ISSUE%'
              OR latest_human.flag LIKE '%UNUSABLE_SEGMENTATION%'
          )
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (issue_n,),
    ).fetchall()

    clean_rows = db.execute(
        """
        WITH latest_human AS (
            SELECT t.line_id, t.flag, t.id
            FROM transcriptions t
            JOIN (
                SELECT line_id, MAX(id) AS latest_id
                FROM transcriptions
                WHERE transcription_type IN ('HUMAN_CORRECTED', 'FLAGGED')
                GROUP BY line_id
            ) latest ON latest.latest_id = t.id
        )
        SELECT
            l.id AS line_id,
            l.crop_image_path,
            p.derived_image_path,
            l.polygon_coords,
            COALESCE(latest_human.flag, '') AS flag
        FROM latest_human
        JOIN lines l ON l.id = latest_human.line_id
        JOIN pages p ON p.id = l.page_id
        WHERE l.skip = 0
          AND l.crop_image_path IS NOT NULL
          AND l.polygon_coords IS NOT NULL
          AND p.derived_image_path IS NOT NULL
          AND (
              latest_human.flag IS NULL
              OR (
                  latest_human.flag NOT LIKE '%SEGMENTATION_ISSUE%'
                  AND latest_human.flag NOT LIKE '%UNUSABLE_SEGMENTATION%'
              )
          )
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (clean_n,),
    ).fetchall()

    samples = []
    for row in issue_rows:
        samples.append({
            "line_id": row["line_id"],
            "crop_path": row["crop_image_path"],
            "flag": row["flag"],
            "bucket": "issue",
        })
    for row in clean_rows:
        samples.append({
            "line_id": row["line_id"],
            "crop_path": row["crop_image_path"],
            "flag": row["flag"],
            "bucket": "clean",
        })

    return samples


@annotate_bp.route("/segment-tuning")
def segment_tuning():
    db = get_db()
    issue_n = _parse_int_param("issue_n", 12, 1, 50)
    clean_n = _parse_int_param("clean_n", 6, 0, 50)

    samples = _sample_segmentation_tuning_lines(db, issue_n=issue_n, clean_n=clean_n)

    defaults = {
        "top_extra": _parse_int_param("top_extra", 8, -40, 120),
        "bottom_extra": _parse_int_param("bottom_extra", 0, -40, 120),
        "left_extra": _parse_int_param("left_extra", 0, -40, 120),
        "right_extra": _parse_int_param("right_extra", 0, -40, 120),
    }

    return render_template(
        "annotate/segment_tuning.html",
        samples=samples,
        defaults=defaults,
        issue_n=issue_n,
        clean_n=clean_n,
    )


@annotate_bp.route("/segment-tuning/api/crop/<int:line_id>")
def api_segment_tuning_crop(line_id: int):
    db = get_db()
    top_extra = _parse_int_param("top_extra", 8, -40, 120)
    bottom_extra = _parse_int_param("bottom_extra", 0, -40, 120)
    left_extra = _parse_int_param("left_extra", 0, -40, 120)
    right_extra = _parse_int_param("right_extra", 0, -40, 120)

    row = db.execute(
        """
        SELECT l.polygon_coords, p.derived_image_path
        FROM lines l
        JOIN pages p ON p.id = l.page_id
        WHERE l.id = ?
          AND l.skip = 0
          AND l.polygon_coords IS NOT NULL
          AND p.derived_image_path IS NOT NULL
        LIMIT 1
        """,
        (line_id,),
    ).fetchone()

    if not row:
        return "", 404

    try:
        points = parse_polygon_coords(row["polygon_coords"])

        page_path = DEFAULT_DB_PATH.parent / row["derived_image_path"]
        if not page_path.exists():
            return "", 404

        with Image.open(page_path) as img:
            w, h = img.size

            bounds = bounds_from_polygon(
                points,
                img_w=w,
                img_h=h,
                top_extra=top_extra,
                bottom_extra=bottom_extra,
                left_extra=left_extra,
                right_extra=right_extra,
            )
            if bounds is None:
                return jsonify({"error": "Invalid crop region with current margins"}), 400

            cx1, cy1, cx2, cy2 = bounds

            cropped = img.crop((cx1, cy1, cx2 + 1, cy2 + 1))
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        current_app.logger.error(f"Segment tuning crop failed for line {line_id}: {e}")
        return "", 500


@annotate_bp.route("/segment-tuning/api/save-config", methods=["POST"])
def api_segment_tuning_save_config():
    db = get_db()
    data = request.get_json(silent=True) or {}

    segmenter = (data.get("segmenter") or "cv_projection").strip() or "cv_projection"
    notes = (data.get("notes") or "").strip()

    params = data.get("params") or {}
    if not isinstance(params, dict):
        return jsonify({"error": "params must be an object"}), 400

    allowed_keys = ("top_extra", "bottom_extra", "left_extra", "right_extra")
    normalized_params: dict[str, int] = {}
    for key in allowed_keys:
        raw = params.get(key, 0)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 0
        normalized_params[key] = max(-40, min(120, value))

    issue_n = data.get("issue_n", 0)
    clean_n = data.get("clean_n", 0)
    try:
        issue_n = max(0, int(issue_n))
    except (TypeError, ValueError):
        issue_n = 0
    try:
        clean_n = max(0, int(clean_n))
    except (TypeError, ValueError):
        clean_n = 0

    raw_ids = data.get("sample_line_ids") or []
    sample_line_ids: list[int] = []
    if isinstance(raw_ids, list):
        for value in raw_ids:
            try:
                sample_line_ids.append(int(value))
            except (TypeError, ValueError):
                continue

    try:
        cursor = db.execute(
            """
            INSERT INTO segmentation_tuning_configs (
                segmenter,
                params_json,
                issue_n,
                clean_n,
                sample_line_ids_json,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                segmenter,
                json.dumps(normalized_params, ensure_ascii=False),
                issue_n,
                clean_n,
                json.dumps(sample_line_ids),
                notes or None,
            ),
        )
        config_id = cursor.lastrowid
        db.commit()

        created = db.execute(
            "SELECT created_at FROM segmentation_tuning_configs WHERE id = ?",
            (config_id,),
        ).fetchone()
        created_at = created["created_at"] if created else None

        return jsonify({
            "success": True,
            "config_id": config_id,
            "created_at": created_at,
            "segmenter": segmenter,
            "params": normalized_params,
        })
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500


@annotate_bp.route("/api/context/<int:line_id>")
def api_context(line_id: int):
    """Return a wider crop of the page image showing context above/below the line."""
    db = get_db()
    row = db.execute("""
        SELECT l.polygon_coords, p.derived_image_path
        FROM lines l
        JOIN pages p ON l.page_id = p.id
        WHERE l.id = ?
    """, (line_id,)).fetchone()

    if not row:
        return "", 404

    try:
        points = parse_polygon_coords(row["polygon_coords"])
        _, y1, _, y2 = polygon_bounds(points)
        line_h = y2 - y1
        padding = max(line_h, 60)  # at least 60px, or one line-height worth of context

        working_dir = DEFAULT_DB_PATH.parent
        page_path = working_dir / row["derived_image_path"]

        with Image.open(page_path) as img:
            img_w, img_h = img.size
            crop_y1 = max(0, y1 - padding)
            crop_y2 = min(img_h, y2 + padding)
            cropped = img.crop((0, crop_y1, img_w, crop_y2))

            # Scale down width to max 2400px to keep response fast
            max_w = 2400
            if cropped.width > max_w:
                ratio = max_w / cropped.width
                new_h = int(cropped.height * ratio)
                cropped = cropped.resize((max_w, new_h), Image.LANCZOS)

            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        current_app.logger.error(f"Context crop failed for line {line_id}: {e}")
        return "", 500


@annotate_bp.route("/")
def index():
    """Main annotation interface - shows one line at a time."""
    db = get_db()
    suggestion_file = request.args.get("suggestions")
    suggestion_files = _list_suggestion_files()
    if suggestion_file and suggestion_file not in suggestion_files:
        suggestion_file = None

    row = _pick_line(db, suggestion_file)
    stats = get_stats(db)

    if not row:
        return render_template(
            "annotate/index.html",
            line=None,
            stats=stats,
            suggestion_mode=bool(suggestion_file),
            suggestion_file=suggestion_file,
            suggestion_files=suggestion_files,
        )
    
    conf = row["confidence"]
    return render_template(
        "annotate/index.html",
        line={
            "id": row["line_id"],
            "crop_path": row["crop_image_path"],
            "ocr_text": row["ocr_text"],
            "confidence": f"{conf:.1%}" if conf is not None else "N/A",
            "model": row["model_version"],
            "ocr_source": row.get("ocr_source", "db_ocr_auto"),
        },
        stats=stats,
        suggestion_mode=bool(suggestion_file),
        suggestion_file=suggestion_file,
        suggestion_files=suggestion_files,
    )


@annotate_bp.route("/api/next")
def api_next():
    """Get next line to annotate (AJAX)."""
    db = get_db()
    suggestion_file = request.args.get("suggestions")
    suggestion_files = _list_suggestion_files()
    if suggestion_file and suggestion_file not in suggestion_files:
        suggestion_file = None

    row = _pick_line(db, suggestion_file)
    
    if not row:
        return jsonify({
            "done": True,
            "stats": get_stats(db),
            "suggestion_mode": bool(suggestion_file),
            "suggestion_file": suggestion_file,
        })
    
    conf = row["confidence"]
    return jsonify({
        "done": False,
        "suggestion_mode": bool(suggestion_file),
        "suggestion_file": suggestion_file,
        "line": {
            "id": row["line_id"],
            "crop_path": row["crop_image_path"],
            "ocr_text": row["ocr_text"],
            "confidence": f"{conf:.1%}" if conf is not None else "N/A",
            "model": row["model_version"],
            "ocr_source": row.get("ocr_source", "db_ocr_auto"),
        },
        "stats": get_stats(db),
    })


@annotate_bp.route("/api/submit", methods=["POST"])
def api_submit():
    """Submit corrected text and/or flags."""
    db = get_db()
    data = request.get_json()
    
    line_id = data.get("line_id")
    corrected_text = _normalize_annotation_text(
        data.get("corrected_text", "").strip()
    )
    flags = data.get("flags", [])  # Array of flag strings
    
    if not line_id:
        return jsonify({"error": "Missing line_id"}), 400

    if not corrected_text and not flags:
        return jsonify({"error": "Must provide corrected text or select a flag"}), 400
    
    # Must have either text or flags
    if not corrected_text and not flags:
        return jsonify({"error": "Must provide corrected text or select a flag"}), 400
    
    try:
        # If text is provided, store as HUMAN_CORRECTED
        if corrected_text:
            db.execute(
                """
                INSERT INTO transcriptions (line_id, transcription_type, text, confidence, created_by, model_version, flag, immutable)
                VALUES (?, 'HUMAN_CORRECTED', ?, 1.0, 'human', 'human', ?, 1)
                """,
                (line_id, corrected_text, ";".join(flags) if flags else None),
            )
        else:
            # If only flags, store as FLAGGED with empty text
            db.execute(
                """
                INSERT INTO transcriptions (line_id, transcription_type, text, confidence, created_by, model_version, flag, immutable)
                VALUES (?, 'FLAGGED', '', 0.0, 'human', 'human', ?, 1)
                """,
                (line_id, ";".join(flags)),
            )
        
        db.commit()
        
        return jsonify({
            "success": True,
            "stats": get_stats(db),
        })
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500


@annotate_bp.route("/review")
def review():
    """View and edit all annotated lines."""
    db = get_db()
    page = int(request.args.get("page", 1))
    per_page = 20
    offset = (page - 1) * per_page
    
    # Get total count (latest human annotation per line)
    count_cursor = db.execute("""
        SELECT COUNT(*) as total
        FROM (
            SELECT line_id, MAX(id) AS latest_id
            FROM transcriptions
            WHERE transcription_type IN ('HUMAN_CORRECTED', 'FLAGGED')
            GROUP BY line_id
        ) latest
    """)
    total = count_cursor.fetchone()["total"]
    total_pages = (total + per_page - 1) // per_page
    
    # Get paginated results (latest human annotation per line)
    cursor = db.execute("""
        WITH latest AS (
            SELECT line_id, MAX(id) AS latest_id
            FROM transcriptions
            WHERE transcription_type IN ('HUMAN_CORRECTED', 'FLAGGED')
            GROUP BY line_id
        )
        SELECT
            t.id as trans_id,
            l.id as line_id,
            l.crop_image_path,
            t.text as corrected_text,
            t.flag,
            t.transcription_type,
            t.created_at,
            (
                SELECT text
                FROM transcriptions
                WHERE line_id = l.id
                  AND transcription_type = 'OCR_AUTO'
                ORDER BY confidence DESC, id DESC
                LIMIT 1
            ) as ocr_text
        FROM latest
        JOIN transcriptions t ON t.id = latest.latest_id
        JOIN lines l ON l.id = t.line_id
        ORDER BY t.created_at DESC, t.id DESC
        LIMIT ? OFFSET ?
    """, (per_page, offset))
    
    items = cursor.fetchall()
    
    return render_template(
        "annotate/review.html",
        items=items,
        page=page,
        total_pages=total_pages,
        total=total,
    )


@annotate_bp.route("/edit/<int:line_id>")
def edit(line_id):
    """Edit an existing annotation."""
    db = get_db()
    
    # Get line info
    cursor = db.execute("""
        SELECT 
            l.id as line_id,
            l.crop_image_path,
            t.id as trans_id,
            t.text as corrected_text,
            t.flag,
            t.transcription_type,
            (
                SELECT text
                FROM transcriptions
                WHERE line_id = l.id
                  AND transcription_type = 'OCR_AUTO'
                ORDER BY confidence DESC, id DESC
                LIMIT 1
            ) as ocr_text
        FROM lines l
        LEFT JOIN transcriptions t ON t.id = (
            SELECT t2.id
            FROM transcriptions t2
            WHERE t2.line_id = l.id
              AND t2.transcription_type IN ('HUMAN_CORRECTED', 'FLAGGED')
            ORDER BY t2.created_at DESC, t2.id DESC
            LIMIT 1
        )
        WHERE l.id = ?
        LIMIT 1
    """, (line_id,))
    
    row = cursor.fetchone()
    
    if not row:
        return "Line not found", 404
    
    return render_template(
        "annotate/edit.html",
        line={
            "id": row["line_id"],
            "crop_path": row["crop_image_path"],
            "corrected_text": row["corrected_text"] or "",
            "flag": row["flag"] or "",
            "ocr_text": row["ocr_text"],
            "trans_id": row["trans_id"],
            "transcription_type": row["transcription_type"],
        },
    )


@annotate_bp.route("/api/update", methods=["POST"])
def api_update():
    """Update an existing annotation."""
    db = get_db()
    data = request.get_json()
    
    line_id = data.get("line_id")
    corrected_text = _normalize_annotation_text(
        data.get("corrected_text", "").strip()
    )
    flags = data.get("flags", [])
    # trans_id is intentionally ignored: GT edits are append-only revisions
    # so immutable history remains intact.
    _ = data.get("trans_id")
    
    if not line_id:
        return jsonify({"error": "Missing line_id"}), 400
    
    try:
        # Insert a new revision (append-only)
        if corrected_text:
            db.execute(
                """
                INSERT INTO transcriptions (line_id, transcription_type, text, confidence, created_by, model_version, flag, immutable)
                VALUES (?, 'HUMAN_CORRECTED', ?, 1.0, 'human', 'human', ?, 1)
                """,
                (line_id, corrected_text, ";".join(flags) if flags else None),
            )
        elif flags:
            db.execute(
                """
                INSERT INTO transcriptions (line_id, transcription_type, text, confidence, created_by, model_version, flag, immutable)
                VALUES (?, 'FLAGGED', '', 0.0, 'human', 'human', ?, 1)
                """,
                (line_id, ";".join(flags)),
            )

        db.commit()

        return jsonify({"success": True})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500


def get_stats(db) -> dict:
    """Get current annotation statistics."""
    total_row = db.execute("""
        SELECT COUNT(DISTINCT line_id) as total
        FROM transcriptions
        WHERE transcription_type = 'OCR_AUTO'
    """).fetchone()

    annotated_row = db.execute("""
        SELECT COUNT(DISTINCT line_id) as annotated
        FROM transcriptions
        WHERE transcription_type IN ('HUMAN_CORRECTED', 'FLAGGED')
    """).fetchone()

    total = total_row["total"] or 0
    annotated = annotated_row["annotated"] or 0
    percent = (annotated / total * 100) if total > 0 else 0

    policy = _load_active_text_policy()
    gt_rows = _get_gt_rows_for_quality(db)
    quality = summarize_text_policy_rows(gt_rows, policy)

    return {
        "total": total,
        "annotated": annotated,
        "remaining": total - annotated,
        "percent": percent,
        "progress": f"{annotated}/{total} ({percent:.1f}%)" if total > 0 else "0/0 (0.0%)",
        "policy": {
            "name": policy.get("name"),
            "version": policy.get("version"),
        },
        "quality": quality,
    }
