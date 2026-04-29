from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pdf_parsing_pipeline import build_default_config, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PDF parsing evaluation pipeline on one or more PDFs."
    )
    parser.add_argument("pdfs", nargs="+", help="PDF file paths to process")
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Skip ground-truth generation and only evaluate existing files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override the number of parallel page workers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_default_config()
    if args.no_gt:
        config.generate_ground_truth = False
    if args.workers is not None:
        config.max_workers = max(1, args.workers)

    results = run_pipeline([Path(pdf) for pdf in args.pdfs], config=config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
