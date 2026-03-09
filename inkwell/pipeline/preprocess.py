"""
Phase 2: Preprocessing

Rotate, deskew, and optionally split double-page source images.
Produces normalized page images ready for segmentation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from inkwell.config import get_root_path, resolve_path
from inkwell.db import get_connection

logger = logging.getLogger(__name__)

# Hardcoded parameters (TODO: move to config if tuning needed)
JPEG_QUALITY = 95
DESKEW_ANGLE_THRESHOLD = 10.0  # Max degrees to correct
HOUGH_THRESHOLD = 100  # Hough line detection threshold
DESKEW_MAX_HORIZONTAL_ANGLE = 20.0
DESKEW_MIN_LINES = 8
DESKEW_STD_THRESHOLD = 3.0
SPLIT_CENTER_WINDOW_RATIO = 0.2


def cleanup_incomplete_pages(conn, working_dir: Path) -> None:
    """
    Clean up incomplete page records from interrupted preprocessing.
    
    For DOUBLE layout images, both left and right pages must exist.
    If only one exists, delete it and clean up its derived image file.
    Also removes orphaned image files without database records.
    """
    derived_dir = working_dir / 'derived_images'
    
    cleaned = 0

    # 0. Remove NO_TEXT pages (these should not be preprocessed at all)
    no_text_rows = conn.execute(
        """
        SELECT p.id, p.derived_image_path
        FROM pages p
        JOIN source_images si ON si.id = p.source_image_id
        WHERE si.layout_type = 'NO_TEXT'
        """
    ).fetchall()

    if no_text_rows:
        for row in no_text_rows:
            rel_path = row["derived_image_path"]
            if rel_path:
                file_path = working_dir / rel_path
                if file_path.exists():
                    file_path.unlink()
                    cleaned += 1
        conn.execute(
            """
            DELETE FROM pages
            WHERE source_image_id IN (
                SELECT id FROM source_images WHERE layout_type = 'NO_TEXT'
            )
            """
        )
        conn.commit()
    
    # 1. Find DOUBLE source images with incomplete page records
    cursor = conn.execute("""
        SELECT si.id as source_image_id,
               GROUP_CONCAT(p.side) as sides,
               GROUP_CONCAT(p.derived_image_path) as paths
        FROM source_images si
        JOIN pages p ON p.source_image_id = si.id
        WHERE si.layout_type = 'DOUBLE'
        GROUP BY si.id
        HAVING COUNT(p.id) != 2
    """)
    
    incomplete = cursor.fetchall()
    
    if incomplete:
        logger.info(f"Cleaning up {len(incomplete)} incomplete "
                    "double-page splits...")
        
        for row in incomplete:
            source_image_id = row['source_image_id']
            paths = row['paths'].split(',') if row['paths'] else []
            
            # Delete page records
            conn.execute("""
                DELETE FROM pages
                WHERE source_image_id = ?
            """, (source_image_id,))
            
            # Delete derived image files
            for rel_path in paths:
                if rel_path:
                    file_path = working_dir / rel_path
                    if file_path.exists():
                        file_path.unlink()
                        cleaned += 1
        
        conn.commit()
    
    # 2. Clean up orphaned files (files without DB records)
    if derived_dir.exists():
        # Get all paths in database
        cursor = conn.execute(
            "SELECT derived_image_path FROM pages "
            "WHERE derived_image_path IS NOT NULL"
        )
        db_paths = {row['derived_image_path'] for row in cursor.fetchall()}
        
        # Check all files in derived_images/
        for file_path in derived_dir.glob('*.jpg'):
            rel_path = f"derived_images/{file_path.name}"
            if rel_path not in db_paths:
                file_path.unlink()
                logger.debug(f"Deleted orphaned file: {file_path.name}")
                cleaned += 1
    
    if cleaned > 0:
        logger.info(f"Cleaned up {cleaned} orphaned files")


def preprocess_all(db_path: str, force: bool = False) -> dict:
    """
    Preprocess all source images with confirmed orientation.
    
    Args:
        db_path: Path to database
        force: If True, reprocess even if already done
    
    Returns:
        Summary dict with counts
    """
    conn = get_connection(db_path)
    root_path = get_root_path(conn)
    working_dir = Path(db_path).parent  # working/ directory
    
    # Clean up incomplete double-page splits from previous interrupted runs
    cleanup_incomplete_pages(conn, working_dir)
    
    # Get source images that need preprocessing
    if force:
        query = """
            SELECT si.id, si.asset_id, si.orientation_confirmed,
                   si.layout_type, a.filename, n.folder_name
            FROM source_images si
            JOIN assets a ON si.asset_id = a.id
            JOIN notebooks n ON a.notebook_id = n.id
            WHERE si.orientation_confirmed IS NOT NULL
              AND si.layout_type != 'NO_TEXT'
            ORDER BY n.id, a.file_order
        """
    else:
        query = """
            SELECT si.id, si.asset_id, si.orientation_confirmed,
                   si.layout_type, a.filename, n.folder_name
            FROM source_images si
            JOIN assets a ON si.asset_id = a.id
            JOIN notebooks n ON a.notebook_id = n.id
            LEFT JOIN pages p ON p.source_image_id = si.id
            WHERE si.orientation_confirmed IS NOT NULL
                            AND si.layout_type != 'NO_TEXT'
              AND p.id IS NULL
            ORDER BY n.id, a.file_order
        """
    
    cursor = conn.execute(query)
    source_images = cursor.fetchall()
    
    if len(source_images) == 0:
        logger.info("No source images to preprocess")
        conn.close()
        return {
            'processed': 0,
            'single_pages': 0,
            'double_pages': 0,
            'errors': 0
        }
    
    logger.info(f"Preprocessing {len(source_images)} source images...")
    
    stats = {
        'processed': 0,
        'single_pages': 0,
        'double_pages': 0,
        'errors': 0
    }
    
    # Process with progress bar
    for row in tqdm(source_images, desc="Preprocessing", unit="image"):
        try:
            preprocess_source_image(
                conn,
                root_path,
                working_dir,
                row,
                force=force,
            )
            stats['processed'] += 1
            
            if row['layout_type'] == 'DOUBLE':
                stats['double_pages'] += 1
            else:
                stats['single_pages'] += 1
            
            # Commit after each image to save progress
            conn.commit()
                
        except KeyboardInterrupt:
            logger.warning("Interrupted by user. Progress saved up to this point.")
            conn.rollback()
            conn.close()
            raise
        except Exception as e:
            logger.error(f"Error preprocessing {row['filename']}: {e}",
                         exc_info=True)
            stats['errors'] += 1
            # Rollback this image's changes but continue
            conn.rollback()
    
    conn.close()
    
    logger.info(f"Preprocessing complete: {stats}")
    return stats


def preprocess_source_image(
    conn,
    root_path: Path,
    working_dir: Path,
    row: dict,
    force: bool = False,
) -> None:
    """
    Preprocess a single source image: rotate, deskew, split if double.
    
    Creates page records in DB with derived_image_path.
    """
    source_image_id = row['id']
    orientation = row['orientation_confirmed']
    layout_type = row['layout_type']

    # Explicitly skip NO_TEXT assets
    if layout_type == 'NO_TEXT':
        return

    # In force mode, clear existing derived rows/files for this source image
    if force:
        delete_existing_pages_for_source_image(conn, working_dir, source_image_id)
    
    # Resolve source image path
    rel_path = f"{row['folder_name']}/{row['filename']}"
    source_path = resolve_path(rel_path, root_path)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source image not found: {source_path}")
    
    logger.debug(f"Processing {source_path.name}: "
                f"orientation={orientation}, layout={layout_type}")
    
    # Load image
    img = cv2.imread(str(source_path))
    if img is None:
        raise ValueError(f"Could not load image: {source_path}")
    
    # Step 1: Rotate
    if orientation != 0:
        img = rotate_image(img, orientation)
    
    # Step 2: Deskew
    img = deskew_image(img)
    
    # Step 3: Split if double page
    if layout_type == 'DOUBLE':
        left_img, right_img = split_double_page(img)
        
        # Save and register left page
        left_path = save_derived_image(source_image_id, 'left',
                                      left_img, working_dir)
        create_page_record(conn, source_image_id, 'left',
                          left_path, 'PAGE')
        
        # Save and register right page
        right_path = save_derived_image(source_image_id, 'right',
                                       right_img, working_dir)
        create_page_record(conn, source_image_id, 'right',
                          right_path, 'PAGE')
        
    else:
        # Single page or no-text
        page_type = 'PAGE' if layout_type == 'SINGLE' else 'COVER_OR_NON_TEXT'
        
        # Save and register
        derived_path = save_derived_image(source_image_id, 'full',
                                         img, working_dir)
        create_page_record(conn, source_image_id, 'full',
                          derived_path, page_type)


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate image by specified angle (0, 90, 180, 270).
    """
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def deskew_image(img: np.ndarray) -> np.ndarray:
    """
    Deskew image using Hough line detection on horizontal lines.
    
    Detects dominant horizontal line angle and rotates to straighten.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    height, width = img.shape[:2]

    # Probabilistic Hough lines tends to be more stable for notebooks/forms
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=max(100, int(width * 0.25)),
        maxLineGap=20,
    )

    if lines is None or len(lines) < DESKEW_MIN_LINES:
        logger.debug("No lines detected, skipping deskew")
        return img
    
    # Collect angles of near-horizontal lines (weighted by line length)
    angles = []
    weights = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 20:
            continue

        angle_deg = float(np.degrees(np.arctan2(dy, dx)))

        # Normalize to [-90, 90]
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180

        if abs(angle_deg) <= DESKEW_MAX_HORIZONTAL_ANGLE:
            angles.append(angle_deg)
            weights.append(length)
    
    if len(angles) < DESKEW_MIN_LINES:
        logger.debug("No horizontal lines found, skipping deskew")
        return img
    
    # Weighted average + spread check to avoid unstable corrections
    weights_arr = np.array(weights, dtype=np.float64)
    angles_arr = np.array(angles, dtype=np.float64)
    avg_angle = float(np.average(angles_arr, weights=weights_arr))
    angle_std = float(np.std(angles_arr))

    if angle_std > DESKEW_STD_THRESHOLD:
        logger.debug(
            f"Deskew unstable (std={angle_std:.2f}°), skipping deskew"
        )
        return img
    
    # Only deskew if angle is significant but not too extreme
    if abs(avg_angle) < 0.5:
        logger.debug(f"Angle too small ({avg_angle:.2f}°), skipping deskew")
        return img
    
    # Rotate in the opposite direction of detected skew
    correction_angle = -avg_angle

    if abs(correction_angle) > DESKEW_ANGLE_THRESHOLD:
        logger.warning(f"Angle too large ({correction_angle:.2f}°), "
                      f"limiting to threshold")
        correction_angle = (
            np.sign(correction_angle) * DESKEW_ANGLE_THRESHOLD
        )
    
    logger.debug(
        f"Deskewing by {correction_angle:.2f}° "
        f"(detected={avg_angle:.2f}°, std={angle_std:.2f}°)"
    )
    
    # Rotate to correct skew
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
    
    # Calculate new bounding box to avoid cropping
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # Adjust transformation matrix for new size
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    
    rotated = cv2.warpAffine(img, matrix, (new_width, new_height),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    
    return rotated


def split_double_page(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a double-page image into left and right pages.
    
    Uses vertical projection near center to find likely gutter.
    Falls back to exact center split if confidence is low.
    """
    _, width = img.shape[:2]
    center = width // 2

    # Search only around the center to avoid splitting on margins
    half_window = int(width * SPLIT_CENTER_WINDOW_RATIO)
    start = max(0, center - half_window)
    end = min(width, center + half_window)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Darkness projection (higher means more ink); gutter should be lighter
    darkness = (255.0 - gray.astype(np.float32)).sum(axis=0)

    if end - start < 10:
        mid = center
    else:
        search_slice = darkness[start:end]
        mid = start + int(np.argmin(search_slice))

        # Guard against pathological splits too close to edge
        min_x = int(width * 0.3)
        max_x = int(width * 0.7)
        if mid < min_x or mid > max_x:
            mid = center
    
    left = img[:, :mid]
    right = img[:, mid:]
    
    return left, right


def save_derived_image(source_image_id: int, side: str, img: np.ndarray,
                       working_dir: Path) -> str:
    """
    Save derived image to working/derived_images/ directory.
    
    Returns relative path for database storage.
    """
    # Use working directory in project root, not source root
    derived_dir = working_dir / 'derived_images'
    derived_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    filename = f"si_{source_image_id}_{side}.jpg"
    output_path = derived_dir / filename
    
    # Save as high-quality JPEG
    cv2.imwrite(str(output_path), img,
               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    
    # Return relative path
    return f"derived_images/{filename}"


def delete_existing_pages_for_source_image(
    conn,
    working_dir: Path,
    source_image_id: int,
) -> None:
    """Delete existing page rows and derived image files for one source image."""
    rows = conn.execute(
        """
        SELECT derived_image_path
        FROM pages
        WHERE source_image_id = ?
        """,
        (source_image_id,),
    ).fetchall()

    for row in rows:
        rel_path = row["derived_image_path"]
        if rel_path:
            file_path = working_dir / rel_path
            if file_path.exists():
                file_path.unlink()

    conn.execute(
        """
        DELETE FROM pages
        WHERE source_image_id = ?
        """,
        (source_image_id,),
    )


def create_page_record(conn, source_image_id: int, side: str,
                       derived_path: str, page_type: str) -> None:
    """
    Create page record in database.
    
    Sets processing_status to 'preprocessed'.
    """
    conn.execute(
        """
        INSERT INTO pages(
            source_image_id,
            side,
            page_type,
            processing_status,
            derived_image_path
        )
        VALUES (?, ?, ?, 'preprocessed', ?)
        ON CONFLICT(source_image_id, side) DO UPDATE SET
            page_type = excluded.page_type,
            processing_status = excluded.processing_status,
            derived_image_path = excluded.derived_image_path
        """,
        (source_image_id, side, page_type, derived_path)
    )
    
    logger.debug(f"Created page record: source_image={source_image_id}, "
                f"side={side}, path={derived_path}")
