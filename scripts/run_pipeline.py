#!/usr/bin/env python3
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inkwell pipeline runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ["ingest", "preprocess", "segment", "ocr", "finetune", "export"]:
        cmd = subparsers.add_parser(command)
        cmd.add_argument("--page", type=int, default=None)
        cmd.add_argument("--force", action="store_true")
        cmd.add_argument("--root", default=None)
        cmd.add_argument("--model", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Pipeline command '{args.command}' is not implemented yet. "
        "Phase 0 provides the CLI contract only."
    )


if __name__ == "__main__":
    main()
