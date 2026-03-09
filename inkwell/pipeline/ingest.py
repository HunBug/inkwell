from __future__ import annotations

import sqlite3
from pathlib import Path

from PIL import Image
import pytesseract


def detect_orientation(image_path: Path) -> int:
    """Detect page orientation using Tesseract OSD. Returns 0, 90, 180, or 270."""
    try:
        img = Image.open(image_path)
        osd = pytesseract.image_to_osd(img)
        
        for line in osd.split('\n'):
            if line.startswith('Rotate:'):
                rotation = int(line.split(':')[1].strip())
                return rotation
        
        return 0
    except Exception as e:
        print(f"Warning: OSD failed for {image_path.name}: {e}")
        return 0


def detect_layout(image_path: Path) -> str:
    """Detect layout type. DOUBLE if width > 1.5 * height, else SINGLE."""
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        if width > 1.5 * height:
            return "DOUBLE"
        return "SINGLE"
    except Exception as e:
        print(f"Warning: Layout detection failed for {image_path.name}: {e}")
        return "SINGLE"


def run_ingest(conn: sqlite3.Connection, root_path: Path) -> dict[str, int]:
    """Run orientation and layout detection on all unprocessed source images."""
    cursor = conn.execute(
        """
        SELECT si.id, si.asset_id, a.filename, n.folder_name
        FROM source_images si
        JOIN assets a ON si.asset_id = a.id
        JOIN notebooks n ON a.notebook_id = n.id
        WHERE si.orientation_detected IS NULL
        AND a.should_ocr = 1
        """
    )
    
    rows = cursor.fetchall()
    
    processed = 0
    double_pages = 0
    rotated_pages = 0
    
    for row in rows:
        source_image_id = row["id"]
        filename = row["filename"]
        folder_name = row["folder_name"]
        
        image_path = root_path / folder_name / filename
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        orientation = detect_orientation(image_path)
        layout = detect_layout(image_path)
        
        conn.execute(
            """
            UPDATE source_images
            SET orientation_detected = ?, layout_type = ?
            WHERE id = ?
            """,
            (orientation, layout, source_image_id),
        )
        
        processed += 1
        if layout == "DOUBLE":
            double_pages += 1
        if orientation != 0:
            rotated_pages += 1
    
    conn.commit()
    
    return {
        "processed": processed,
        "double_pages": double_pages,
        "rotated_pages": rotated_pages,
    }
