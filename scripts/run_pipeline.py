#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inkwell pipeline runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ["ingest", "preprocess", "segment", "ocr", "finetune", "export"]:
        cmd = subparsers.add_parser(command)
        cmd.add_argument("--page", type=int, default=None)
        cmd.add_argument("--force", action="store_true")
        cmd.add_argument("--root", default=None)
        cmd.add_argument("--model", default=None)
        if command == "ocr":
            cmd.add_argument(
                "--limit",
                type=int,
                default=None,
                help="Maximum number of lines to OCR in this run",
            )
            cmd.add_argument(
                "--langs",
                default="hu,en",
                help="Comma-separated OCR languages for backend (default: hu,en)",
            )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Get database path
    from inkwell.db import get_connection
    db_path = str(PROJECT_ROOT / "working" / "inkwell.db")
    
    if args.command == "ingest":
        from inkwell.config import get_root_path
        from inkwell.pipeline.ingest import run_ingest
        
        conn = get_connection(db_path)
        root_path = get_root_path(conn, args.root)
        
        print("Running orientation and layout detection...")
        stats = run_ingest(conn, root_path)
        
        print(f"\nIngest complete:")
        print(f"  Processed: {stats['processed']} images")
        print(f"  Double-page layouts: {stats['double_pages']}")
        print(f"  Rotated pages: {stats['rotated_pages']}")
        
    elif args.command == "preprocess":
        from inkwell.pipeline.preprocess import preprocess_all
        
        print("Running preprocessing (rotate, deskew, split)...")
        stats = preprocess_all(db_path, force=args.force)
        
        print(f"\nPreprocessing complete:")
        print(f"  Processed: {stats['processed']} source images")
        print(f"  Single pages: {stats['single_pages']}")
        print(f"  Double pages: {stats['double_pages']} (split into {stats['double_pages']*2} pages)")
        print(f"  Errors: {stats['errors']}")
        
    elif args.command == "segment":
        from inkwell.pipeline.segment import segment_all
        
        method = args.model or 'cv_projection'
        print(f"Running line segmentation (method={method})...")
        stats = segment_all(db_path, force=args.force, method=method)
        
        print(f"\nSegmentation complete:")
        print(f"  Processed: {stats['processed']} pages")
        print(f"  Total lines extracted: {stats['total_lines']}")
        print(f"  Errors: {stats['errors']}")

    elif args.command == "ocr":
        from inkwell.pipeline.ocr import run_ocr

        model = args.model or "easyocr"
        languages = [part.strip() for part in args.langs.split(",") if part.strip()]
        print(
            "Running OCR "
            f"(model={model}, langs={languages}, page={args.page}, "
            f"limit={args.limit}, force={args.force})..."
        )
        stats = run_ocr(
            db_path,
            model=model,
            languages=languages,
            page=args.page,
            limit=args.limit,
            force=args.force,
        )

        print("\nOCR complete:")
        print(f"  Processed: {stats['processed']} lines")
        print(f"  Empty text: {stats['empty']}")
        print(f"  Errors: {stats['errors']}")
        
    else:
        print(
            f"Pipeline command '{args.command}' is not implemented yet. "
            "Additional stages coming in later phases."
        )


if __name__ == "__main__":
    main()
