from __future__ import annotations

import io
import json

from flask import Blueprint, render_template, request, jsonify, current_app, g, send_file
from pathlib import Path
from PIL import Image

from inkwell.db import get_connection, DEFAULT_DB_PATH


annotate_bp = Blueprint("annotate", __name__, url_prefix="/annotate")


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
    return g.db


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
        return _get_next_suggested_unannotated_line(db, suggestion_filename)
    return _get_random_unannotated_line(db)


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
        coords = json.loads(row["polygon_coords"])  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        ys = [pt[1] for pt in coords]
        y1, y2 = min(ys), max(ys)
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

    return {
        "total": total,
        "annotated": annotated,
        "remaining": total - annotated,
        "percent": percent,
        "progress": f"{annotated}/{total} ({percent:.1f}%)" if total > 0 else "0/0 (0.0%)",
    }
