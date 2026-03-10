#!/usr/bin/env python3
"""
Generate an HTML visual sample of OCR results.

Shows random line crops with their OCR text, confidence, and metadata
for quick quality assessment.
"""
from __future__ import annotations

import argparse
import base64
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inkwell.db import get_connection


def generate_sample_html(
    db_path: str,
    output_path: Path,
    sample_size: int = 30,
    transcription_type: str = "OCR_AUTO",
    model_key: str | None = None,
) -> None:
    """Generate HTML page with random OCR samples."""
    conn = get_connection(db_path)
    working_dir = Path(db_path).parent
    line_crops_dir = working_dir / "line_crops"

    # Get random sample of OCR'd lines
    query = """
        SELECT
            t.line_id,
            t.text,
            t.confidence,
            t.model_version,
            t.created_by,
            t.created_at,
            l.page_id,
            l.line_order,
            l.crop_image_path,
            l.segmentation_confidence
        FROM transcriptions t
        JOIN lines l ON l.id = t.line_id
        WHERE t.transcription_type = ?
    """
    params: list[object] = [transcription_type]

    if model_key:
        query += " AND t.created_by = ?"
        params.append(model_key)

    query += " ORDER BY RANDOM() LIMIT ?"
    params.append(sample_size)

    rows = conn.execute(query, params).fetchall()

    if not rows:
        print(f"No transcriptions found with type '{transcription_type}'")
        return

    print(f"Sampled {len(rows)} OCR results")

    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OCR Sample Review</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .stats {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample {
            background: white;
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 0.9em;
            color: #666;
        }
        .meta {
            display: flex;
            gap: 15px;
        }
        .meta-item {
            display: flex;
            gap: 5px;
        }
        .meta-label {
            font-weight: bold;
        }
        .confidence {
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .conf-high { background: #4CAF50; color: white; }
        .conf-med { background: #FFC107; color: black; }
        .conf-low { background: #F44336; color: white; }
        .image-container {
            background: #fafafa;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            text-align: center;
        }
        .line-image {
            max-width: 100%;
            border: 1px solid #ddd;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
        .ocr-text {
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
            padding: 10px;
            background: #f9f9f9;
            border-left: 4px solid #4CAF50;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .empty-text {
            color: #999;
            font-style: italic;
        }
    </style>
</head>
<body>
    <h1>📝 OCR Quality Sample</h1>
    <div class="stats">
        <strong>Sample size:</strong> """ + str(len(rows)) + """ lines (random)<br>
        <strong>Transcription type:</strong> """ + transcription_type + """<br>
        <strong>Database:</strong> """ + str(db_path) + """<br>
        <strong>Model filter:</strong> """ + (model_key or "(all models)") + """
    </div>
"""
    ]

    for row in rows:
        line_id = row["line_id"]
        text = row["text"] or ""
        confidence = row["confidence"]
        model_version = row["model_version"]
        created_by = row["created_by"]
        page_id = row["page_id"]
        line_order = row["line_order"]
        crop_path = row["crop_image_path"]

        # Confidence badge
        if confidence is not None:
            if confidence >= 0.7:
                conf_class = "conf-high"
            elif confidence >= 0.4:
                conf_class = "conf-med"
            else:
                conf_class = "conf-low"
            conf_display = f'<span class="confidence {conf_class}">{confidence:.2f}</span>'
        else:
            conf_display = '<span class="confidence conf-low">N/A</span>'

        # Load and encode image
        image_path = line_crops_dir / crop_path
        if image_path.exists():
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            img_tag = f'<img class="line-image" src="data:image/jpeg;base64,{img_data}" alt="Line crop">'
        else:
            img_tag = f'<div style="color: red;">Image not found: {crop_path}</div>'

        # Text display
        if text.strip():
            text_display = f'<div class="ocr-text">{text}</div>'
        else:
            text_display = '<div class="ocr-text empty-text">(empty OCR result)</div>'

        html_parts.append(
            f"""
    <div class="sample">
        <div class="sample-header">
            <div class="meta">
                <div class="meta-item">
                    <span class="meta-label">Page:</span> {page_id}
                </div>
                <div class="meta-item">
                    <span class="meta-label">Line:</span> {line_order}
                </div>
                <div class="meta-item">
                    <span class="meta-label">Line ID:</span> {line_id}
                </div>
                <div class="meta-item">
                    <span class="meta-label">Model:</span> {model_version}
                </div>
                <div class="meta-item">
                    <span class="meta-label">Engine:</span> {created_by}
                </div>
            </div>
            <div>
                {conf_display}
            </div>
        </div>
        <div class="image-container">
            {img_tag}
        </div>
        {text_display}
    </div>
"""
        )

    html_parts.append(
        """
</body>
</html>
"""
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    print(f"\nGenerated sample: {output_path}")
    print(f"Open in browser: file://{output_path.absolute()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OCR quality sample HTML")
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "working" / "inkwell.db"),
        help="Path to database",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "working" / "ocr_sample.html"),
        help="Output HTML path",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=30,
        help="Number of samples to include",
    )
    parser.add_argument(
        "--type",
        default="OCR_AUTO",
        help="Transcription type to sample",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional OCR engine filter (e.g. easyocr, trocr)",
    )

    args = parser.parse_args()

    generate_sample_html(
        db_path=args.db,
        output_path=Path(args.output),
        sample_size=args.size,
        transcription_type=args.type,
        model_key=args.model,
    )


if __name__ == "__main__":
    main()
