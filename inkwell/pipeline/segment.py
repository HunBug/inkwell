"""
Phase 3: Line Segmentation

Detect and extract individual text lines from preprocessed page images.
Creates line crops and database records for subsequent OCR.

Segmentation can be swapped between different algorithms via the
SEGMENTATION_METHOD config setting (currently only 'cv_projection' implemented).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from inkwell.config import get_root_path, resolve_path
from inkwell.db import get_connection

logger = logging.getLogger(__name__)

# Hardcoded parameters (TODO: move to config/database settings)
JPEG_QUALITY = 95

# CV projection parameters
MIN_LINE_HEIGHT = 10  # pixels
MAX_LINE_GAP = 5      # pixels - merge lines closer than this
MIN_LINE_WIDTH_RATIO = 0.1  # reject lines narrower than 10% of page width
PROJECTION_THRESHOLD_PERCENTILE = 50  # threshold for binarizing projection


def segment_lines_cv_projection(
    image: np.ndarray,
    min_height: int = MIN_LINE_HEIGHT,
    max_gap: int = MAX_LINE_GAP,
    min_width_ratio: float = MIN_LINE_WIDTH_RATIO,
) -> list[dict[str, Any]]:
    """
    Segment text lines using horizontal projection profile.
    
    Returns list of line dictionaries with:
      - bbox: [x, y, w, h]
      - polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (clockwise from top-left)
      - confidence: float (0-1)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Binarize
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Horizontal projection profile (sum of foreground pixels per row)
    projection = np.sum(binary, axis=1) / 255.0  # normalize
    
    # Threshold to find text rows
    threshold = np.percentile(projection[projection > 0], PROJECTION_THRESHOLD_PERCENTILE)
    text_rows = projection > threshold
    
    # Find contiguous runs of text rows
    lines = []
    in_line = False
    line_start = 0
    
    for i, is_text in enumerate(text_rows):
        if is_text and not in_line:
            line_start = i
            in_line = True
        elif not is_text and in_line:
            line_end = i - 1
            if line_end - line_start >= min_height:
                lines.append((line_start, line_end))
            in_line = False
    
    # Handle last line if it extends to the end
    if in_line and len(text_rows) - line_start >= min_height:
        lines.append((line_start, len(text_rows) - 1))
    
    # Merge lines that are close together
    merged_lines = []
    for line_start, line_end in lines:
        if merged_lines and line_start - merged_lines[-1][1] <= max_gap:
            merged_lines[-1] = (merged_lines[-1][0], line_end)
        else:
            merged_lines.append((line_start, line_end))
    
    # Extract bounding boxes with horizontal extent
    page_width = image.shape[1]
    min_width = int(page_width * min_width_ratio)
    
    result = []
    for line_start, line_end in merged_lines:
        # Find horizontal extent using column-wise OR across the line's rows
        line_slice = binary[line_start:line_end+1, :]
        col_projection = np.any(line_slice > 0, axis=0)
        
        # Find leftmost and rightmost text columns
        text_cols = np.where(col_projection)[0]
        if len(text_cols) == 0:
            continue
        
        x1 = int(text_cols[0])
        x2 = int(text_cols[-1])
        width = x2 - x1 + 1
        
        if width < min_width:
            continue
        
        # Add generous vertical margins to capture ascenders/descenders.
        # Top margin is slightly larger than bottom to reduce accent clipping.
        margin_top = 24
        margin_bottom = 19
        margin_x = 5
        y_start = max(0, line_start - margin_top)
        y_end = min(image.shape[0] - 1, line_end + margin_bottom)
        x_start = max(0, x1 - margin_x)
        x_end = min(image.shape[1] - 1, x2 + margin_x)
        
        height = y_end - y_start + 1
        width = x_end - x_start + 1
        
        # Confidence based on line height (simple heuristic)
        confidence = min(1.0, height / 60.0)  # Assume ~60px is "normal" line height
        
        result.append({
            'bbox': [x_start, y_start, width, height],
            'polygon': [
                [x_start, y_start],
                [x_start + width, y_start],
                [x_start + width, y_start + height],
                [x_start, y_start + height],
            ],
            'confidence': confidence,
        })
    
    return result


def segment_page(
    page_id: int,
    image_path: Path,
    output_dir: Path,
    method: str = 'cv_projection',
) -> dict[str, Any]:
    """
    Segment a single page image into lines.
    
    Returns:
      - lines: list of line dicts (bbox, polygon, confidence)
      - segmentation_type: str
      - model_version: str (e.g., 'cv_projection_v1')
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    if method == 'cv_projection':
        lines = segment_lines_cv_projection(image)
        model_version = 'cv_projection_v1'
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    # Save line crops
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, line in enumerate(lines):
        bbox = line['bbox']
        x, y, w, h = bbox
        crop = image[y:y+h, x:x+w]
        
        crop_filename = f"page_{page_id:06d}_line_{i:03d}.jpg"
        crop_path = output_dir / crop_filename
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        
        line['crop_path'] = crop_filename
    
    return {
        'lines': lines,
        'segmentation_type': method.upper(),
        'model_version': model_version,
    }


def save_segmentation_to_db(
    conn,
    page_id: int,
    segmentation_result: dict[str, Any],
) -> None:
    """
    Save segmentation and line records to database.
    Overwrites existing segmentation if force mode.
    """
    lines = segmentation_result['lines']
    segmentation_type = segmentation_result['segmentation_type']
    model_version = segmentation_result['model_version']
    
    # Create segmentation record
    cursor = conn.execute(
        """
        INSERT INTO segmentations (page_id, segmentation_type, model_version)
        VALUES (?, ?, ?)
        """,
        (page_id, segmentation_type, model_version),
    )
    segmentation_id = cursor.lastrowid
    
    # Create line records
    for i, line in enumerate(lines):
        polygon_json = json.dumps(line['polygon'])
        conn.execute(
            """
            INSERT INTO lines (
                page_id, segmentation_id, line_order,
                polygon_coords, crop_image_path, segmentation_confidence
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                page_id,
                segmentation_id,
                i,
                polygon_json,
                line['crop_path'],
                line['confidence'],
            ),
        )
    
    # Update page status
    conn.execute(
        """
        UPDATE pages
        SET processing_status = 'segmented'
        WHERE id = ?
        """,
        (page_id,),
    )
    
    conn.commit()


def cleanup_existing_segmentation(conn, page_id: int, output_dir: Path) -> None:
    """
    Remove existing segmentation and line records/files for a page.
    Used in force mode.
    """
    # Get all line crop paths before deleting
    rows = conn.execute(
        """
        SELECT crop_image_path FROM lines
        WHERE page_id = ?
        """,
        (page_id,),
    ).fetchall()
    
    # Delete files
    for row in rows:
        if row['crop_image_path']:
            crop_path = output_dir / row['crop_image_path']
            if crop_path.exists():
                crop_path.unlink()
    
    # Delete database records (must respect foreign key constraints)
    # Order: transcriptions -> lines -> segmentations
    conn.execute(
        """
        DELETE FROM transcriptions
        WHERE line_id IN (SELECT id FROM lines WHERE page_id = ?)
        """,
        (page_id,),
    )
    conn.execute("DELETE FROM lines WHERE page_id = ?", (page_id,))
    conn.execute("DELETE FROM segmentations WHERE page_id = ?", (page_id,))
    conn.commit()


def segment_all(
    db_path: str,
    force: bool = False,
    method: str = 'cv_projection',
) -> dict[str, int]:
    """
    Segment all preprocessed pages into text lines.
    
    Args:
        db_path: Path to SQLite database
        force: If True, reprocess already-segmented pages
        method: Segmentation algorithm to use ('cv_projection', etc.)
    
    Returns:
        Statistics dictionary with counts
    """
    conn = get_connection(db_path)
    working_dir = Path(db_path).parent
    output_dir = working_dir / 'line_crops'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find pages that need segmentation
    if force:
        query = """
            SELECT id, derived_image_path
            FROM pages
            WHERE processing_status IN ('preprocessed', 'segmented')
              AND derived_image_path IS NOT NULL
            ORDER BY id
        """
    else:
        query = """
            SELECT id, derived_image_path
            FROM pages
            WHERE processing_status = 'preprocessed'
              AND derived_image_path IS NOT NULL
            ORDER BY id
        """
    
    pages = conn.execute(query).fetchall()
    
    if not pages:
        print("No pages to segment.")
        return {
            'processed': 0,
            'total_lines': 0,
            'errors': 0,
        }
    
    stats = {
        'processed': 0,
        'total_lines': 0,
        'errors': 0,
    }
    
    try:
        for row in tqdm(pages, desc="Segmenting pages", unit="page"):
            page_id = row['id']
            derived_path_rel = row['derived_image_path']
            
            if not derived_path_rel:
                logger.warning(f"Page {page_id} has no derived_image_path, skipping")
                continue
            
            image_path = working_dir / derived_path_rel
            
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                stats['errors'] += 1
                continue
            
            try:
                # Clean up existing segmentation if force mode
                if force:
                    cleanup_existing_segmentation(conn, page_id, output_dir)
                
                # Run segmentation
                result = segment_page(page_id, image_path, output_dir, method=method)
                
                # Save to database
                save_segmentation_to_db(conn, page_id, result)
                
                stats['processed'] += 1
                stats['total_lines'] += len(result['lines'])
                
            except Exception as e:
                logger.error(f"Failed to segment page {page_id}: {e}")
                stats['errors'] += 1
                conn.rollback()
    
    except KeyboardInterrupt:
        print("\n\nSegmentation interrupted by user. Progress has been saved.")
        conn.commit()
    
    return stats
