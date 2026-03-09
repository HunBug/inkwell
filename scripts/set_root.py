#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inkwell.config import set_root_path
from inkwell.db import get_connection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update root_path in DB config")
    parser.add_argument("root_path", help="New root path")
    parser.add_argument("--db", default=None, help="Override DB path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn = get_connection(args.db)
    root = set_root_path(conn, args.root_path)
    print(f"Updated root_path: {root}")


if __name__ == "__main__":
    main()
