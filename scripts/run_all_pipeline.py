#!/usr/bin/env python3
"""Run the full GDPR enforcement data pipeline end-to-end."""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar


PROJECT_ROOT = Path(__file__).resolve().parent.parent
T = TypeVar("T")


def load_module(name: str, relative_path: str):
    """Dynamically load a phase module by path."""

    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_path(path: Path) -> Path:
    """Resolve paths relative to the project root."""

    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_with_guard(label: str, func: Callable[..., T], *args, **kwargs) -> T:
    """Execute a callable and re-wrap common IO errors with actionable messages."""

    try:
        return func(*args, **kwargs)
    except FileNotFoundError as exc:
        missing = getattr(exc, "filename", None) or str(exc)
        raise SystemExit(
            f"[{label}] Missing file: {missing}. Re-run the preceding phase or update the path."
        ) from exc
    except Exception as exc:
        raise SystemExit(f"[{label}] Failed with error: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phases 1-4 sequentially using in-repo helpers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Path to an AI responses text file (Answer N format).",
    )
    parser.add_argument(
        "--input-text",
        type=str,
        help="Raw AI responses text. When provided, the script writes it to a temporary file before processing.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Root directory for phase output folders.",
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip Phase 4 enrichment (Phases 1-3 still run).",
    )
    parser.add_argument(
        "--keep-temp-input",
        action="store_true",
        help="Preserve the temporary file created from --input-text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_file and args.input_text:
        raise SystemExit("Provide either --input-file or --input-text, not both.")

    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    temp_input_path: Optional[Path] = None

    if args.input_text:
        tmp_dir = output_root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, dir=tmp_dir) as tmp_file:
            tmp_file.write(args.input_text)
            temp_input_path = Path(tmp_file.name)
        input_path = temp_input_path
        print(f"Created temporary AI response file at {input_path}")
    elif args.input_file:
        input_path = resolve_path(args.input_file)
    else:
        input_path = PROJECT_ROOT / "raw_data" / "AI_analysis" / "AI-responses.txt"

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    phase1_module = load_module("phase1_parser", "scripts/1_parse_ai_responses.py")
    phase2_module = load_module("phase2_validator", "scripts/2_validate_dataset.py")
    phase3_module = load_module("phase3_repair", "scripts/3_repair_data_errors.py")

    phase1_dir = output_root / "phase1_extraction"
    phase1_results: Dict[str, Any] = run_with_guard(
        "Phase 1",
        phase1_module.run_phase1,
        input_path,
        output_dir=phase1_dir,
        verbose=True,
    )
    main_dataset_path = Path(phase1_results["main_csv"])

    phase2_dir = output_root / "phase2_validation"
    phase2_validated_path = phase2_dir / "validated_data.csv"
    phase2_errors_path = phase2_dir / "validation_errors.csv"
    phase2_report_path = phase2_dir / "validation_report.txt"
    phase2_results: Dict[str, Any] = run_with_guard(
        "Phase 2",
        phase2_module.run_phase2,
        main_dataset_path,
        output_validated=phase2_validated_path,
        output_errors=phase2_errors_path,
        output_report=phase2_report_path,
        verbose=True,
    )

    error_csv_path = phase2_results.get("errors_csv")
    if error_csv_path is None or not Path(error_csv_path).exists():
        print("No validation errors detected; skipping Phase 3 automatic repairs.")
        repaired_dataset_path = main_dataset_path
    else:
        phase3_dir = output_root / "phase3_repair"
        repaired_dataset_path = phase3_dir / "repaired_dataset.csv"
        repair_log_path = phase3_dir / "repair_log.txt"
        phase3_results: Dict[str, Any] = run_with_guard(
            "Phase 3",
            phase3_module.run_phase3,
            main_dataset_path,
            Path(error_csv_path),
            output_file=repaired_dataset_path,
            log_file=repair_log_path,
            verbose=True,
        )
        repaired_dataset_path = Path(phase3_results["repaired_csv"])

    phase3_dir = output_root / "phase3_repair"
    revalidation_results = run_with_guard(
        "Phase 2 revalidation",
        phase2_module.run_phase2,
        repaired_dataset_path,
        output_validated=phase3_dir / "repaired_dataset_validated.csv",
        output_errors=phase3_dir / "repaired_dataset_validation_errors.csv",
        output_report=phase3_dir / "repaired_dataset_validation_report.txt",
        verbose=True,
    )

    if not args.skip_enrichment:
        phase4_module = load_module("phase4_enrich", "scripts/4_enrich_prepare_outputs.py")
        enrichment_dir = output_root / "phase4_enrichment"
        enrichment_dir.mkdir(parents=True, exist_ok=True)
        fx_table = PROJECT_ROOT / "raw_data" / "reference" / "fx_rates.csv"
        hicp_table = PROJECT_ROOT / "raw_data" / "reference" / "hicp_ea19.csv"
        context_taxonomy = PROJECT_ROOT / "raw_data" / "reference" / "context_taxonomy.csv"
        region_map = PROJECT_ROOT / "raw_data" / "reference" / "region_map.csv"
        run_with_guard(
            "Phase 4",
            phase4_module.enrich_dataset,
            repaired_dataset_path,
            enrichment_dir,
            fx_table,
            hicp_table,
            context_taxonomy,
            region_map,
        )
        print(f"✓ Phase 4 enrichment complete. Outputs written to {enrichment_dir}")
    else:
        print("Phase 4 enrichment skipped by user request.")

    if temp_input_path and not args.keep_temp_input:
        temp_input_path.unlink(missing_ok=True)
        print(f"Removed temporary input file {temp_input_path}")

    print("\nPipeline summary:")
    print(f"  Phase 1 main dataset: {main_dataset_path}")
    validated_csv = phase2_results.get("validated_csv")
    if validated_csv:
        print(f"  Phase 2 validated subset: {validated_csv}")
    errors_csv = phase2_results.get("errors_csv")
    if errors_csv:
        print(f"  Phase 2 error ledger: {errors_csv}")
    print(f"  Phase 3 revalidated dataset: {revalidation_results.get('validated_csv')}")
    if not args.skip_enrichment:
        print(f"  Phase 4 directory: {output_root / 'phase4_enrichment'}")


if __name__ == "__main__":
    main()
