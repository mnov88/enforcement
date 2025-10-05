"""CLI wrapper for Research Task 3 (harmonization & heterogeneity)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_tasks import task3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Research Task 3 harmonization diagnostics."
    )
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
        help="Optional override for the output directory (defaults to outputs/research_tasks/task3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task3.run(output_dir=args.output_dir, data_path=args.data_path)


if __name__ == "__main__":
    main()
