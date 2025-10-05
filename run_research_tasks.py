"""Orchestrator for sequential execution of research tasks."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable

from research_tasks import task0, task1, task2

TaskRunner = Callable[..., Path]

TASK_SEQUENCE: Dict[str, TaskRunner] = {
    "task0": task0.run,
    "task1": task1.run,
    "task2": task2.run,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute research tasks in sequence.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASK_SEQUENCE.keys()),
        default=list(TASK_SEQUENCE.keys()),
        help="Subset of tasks to run (defaults to task0 task1).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional override for the enriched master dataset path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for task outputs (defaults to outputs/research_tasks).",
    )
    return parser.parse_args()


def run_selected_tasks(task_names: Iterable[str], *, data_path: Path | None, output_root: Path | None) -> None:
    for name in task_names:
        runner = TASK_SEQUENCE[name]
        print(f"Running {name}…")
        output_dir = (output_root / name) if output_root is not None else None
        runner(output_dir=output_dir, data_path=data_path)
    print("All requested tasks completed.")


def main() -> None:
    args = parse_args()
    run_selected_tasks(args.tasks, data_path=args.data_path, output_root=args.output_root)


if __name__ == "__main__":
    main()
