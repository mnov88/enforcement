"""Orchestrator for Research Task 5 end-to-end execution."""
from __future__ import annotations

import argparse
from pathlib import Path

from research_tasks import task5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Research Task 5 pipeline.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional override for the enriched master dataset path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for the task output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task5.run(output_dir=args.output_dir, data_path=args.data_path)


if __name__ == "__main__":
    main()
