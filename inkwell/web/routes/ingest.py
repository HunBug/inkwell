from __future__ import annotations

from flask import Blueprint, render_template, request, jsonify, current_app, g, send_file
from pathlib import Path

from inkwell.config import get_root_path, resolve_path
from inkwell.db import get_connection


ingest_bp = Blueprint("ingest", __name__, url_prefix="/ingest")


def get_db():
    """Get database connection for current request."""
    if "db" not in g:
        g.db = get_connection(current_app.config.get("DB_PATH"))
    return g.db


@ingest_bp.route("/")
def list_images():
    db = get_db()
    root_path = get_root_path(db)
    
    filter_by = request.args.get("filter", "pending")
    page = int(request.args.get("page", 1))
    
    # Different page sizes for different filters
    if filter_by == "pending":
        per_page = 10  # Quick review for pending
    else:
        per_page = 50  # Show more for reviewed images
    
    offset = (page - 1) * per_page
    
    # Build WHERE clause based on filter
    where_clause = """
        FROM source_images si
        JOIN assets a ON si.asset_id = a.id
        JOIN notebooks n ON a.notebook_id = n.id
    """
    
    if filter_by == "pending":
        where_clause += " WHERE si.orientation_confirmed IS NULL"
    elif filter_by == "single":
        where_clause += " WHERE si.layout_type = 'SINGLE' AND si.orientation_confirmed IS NOT NULL"
    elif filter_by == "double":
        where_clause += " WHERE si.layout_type = 'DOUBLE' AND si.orientation_confirmed IS NOT NULL"
    elif filter_by == "no_text":
        where_clause += " WHERE si.layout_type = 'NO_TEXT' AND si.orientation_confirmed IS NOT NULL"
    
    order_clause = " ORDER BY n.id, a.file_order"
    
    # Get total count
    count_cursor = db.execute(f"SELECT COUNT(*) as cnt {where_clause}")
    total_count = count_cursor.fetchone()["cnt"]
    total_pages = (total_count + per_page - 1) // per_page
    
    cursor = db.execute(
        f"""
        SELECT 
            si.id,
            si.orientation_detected,
            si.orientation_confirmed,
            si.layout_type,
            a.filename,
            a.asset_type,
            a.should_ocr,
            n.folder_name,
            n.label as notebook_label
        {where_clause}
        {order_clause}
        LIMIT ? OFFSET ?
        """,
        (per_page, offset),
    )
    
    images = cursor.fetchall()
    
    return render_template(
        "ingest.html",
        images=images,
        filter_by=filter_by,
        page=page,
        total_pages=total_pages,
        total_count=total_count,
        per_page=per_page,
    )


@ingest_bp.route("/<int:source_image_id>/confirm", methods=["POST"])
def confirm_image(source_image_id: int):
    db = get_db()
    data = request.get_json()
    
    orientation = data.get("orientation")
    layout = data.get("layout")
    should_ocr = data.get("should_ocr", 1)
    
    print(f"DEBUG: Confirming image {source_image_id}: orientation={orientation}, layout={layout}, should_ocr={should_ocr}")
    
    # Convert orientation to int
    try:
        orientation = int(orientation)
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": f"Invalid orientation: {orientation}"}), 400
    
    db.execute(
        """
        UPDATE source_images
        SET orientation_confirmed = ?, layout_type = ?
        WHERE id = ?
        """,
        (orientation, layout, source_image_id),
    )
    
    asset_id = db.execute(
        "SELECT asset_id FROM source_images WHERE id = ?",
        (source_image_id,),
    ).fetchone()["asset_id"]
    
    db.execute(
        "UPDATE assets SET should_ocr = ? WHERE id = ?",
        (should_ocr, asset_id),
    )
    
    db.commit()
    
    return jsonify({"status": "ok"})


@ingest_bp.route("/bulk-confirm", methods=["POST"])
def bulk_confirm():
    db = get_db()
    data = request.get_json()
    ids = data.get("ids", [])
    
    for source_image_id in ids:
        row = db.execute(
            """
            SELECT orientation_detected, layout_type
            FROM source_images WHERE id = ?
            """,
            (source_image_id,),
        ).fetchone()
        
        if row:
            db.execute(
                """
                UPDATE source_images
                SET orientation_confirmed = ?
                WHERE id = ?
                """,
                (row["orientation_detected"], source_image_id),
            )
    
    db.commit()
    
    return jsonify({"status": "ok", "count": len(ids)})


@ingest_bp.route("/image/<int:source_image_id>")
def serve_image(source_image_id: int):
    """Serve the actual image file."""
    db = get_db()
    root_path = get_root_path(db)
    
    row = db.execute(
        """
        SELECT n.folder_name, a.filename
        FROM source_images si
        JOIN assets a ON si.asset_id = a.id
        JOIN notebooks n ON a.notebook_id = n.id
        WHERE si.id = ?
        """,
        (source_image_id,),
    ).fetchone()
    
    if not row:
        return "Image not found", 404
    
    # Construct relative path from folder_name/filename
    rel_path = f"{row['folder_name']}/{row['filename']}"
    image_path = resolve_path(rel_path, root_path)
    
    if not image_path.exists():
        return "Image file not found", 404
    
    return send_file(image_path)
