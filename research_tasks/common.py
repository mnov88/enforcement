"""Shared helpers for GDPR enforcement research tasks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

DATA_PATH = Path("outputs/phase4_enrichment/1_enriched_master.csv")
OUTPUT_ROOT = Path("outputs/research_tasks")
RANDOM_SEED = 42

# Art. 58 measure boolean columns used throughout the research tasks.
MEASURE_COLUMNS: Sequence[str] = (
    "a45_warning_issued_bool",
    "a46_reprimand_issued_bool",
    "a47_comply_data_subject_order_bool",
    "a48_compliance_order_bool",
    "a49_breach_communication_order_bool",
    "a50_erasure_restriction_order_bool",
    "a51_certification_withdrawal_bool",
    "a52_data_flow_suspension_bool",
)

# Boolean helper columns that should use pandas' nullable Boolean dtype.
BOOLEAN_COLUMNS: Sequence[str] = (
    "fine_imposed_bool",
    "measure_any_bool",
    "fine_present",
    "has_complaint_bool",
    "official_audit_bool",
    "art33_discussed_bool",
    "art33_breached_bool",
    "oss_case_bool",
    "oss_role_lead_bool",
    "oss_role_concerned_bool",
    "art83_systematic_bool",
)

SEVERITY_FLAGS: Sequence[str] = (
    "breach_has_art5",
    "breach_has_art6",
    "breach_has_art32",
    "breach_has_art33",
)

ART83_FACTOR_COLUMNS: Sequence[str] = (
    "a59_nature_gravity_duration",
    "a60_intentional_negligent",
    "a61_mitigate_damage_actions",
    "a62_technical_org_measures",
    "a63_previous_infringements",
    "a64_cooperation_authority",
    "a65_data_categories_affected",
    "a66_infringement_became_known",
    "a67_prior_orders_compliance",
    "a68_codes_certification",
    "a69_other_factors",
)

ART83_SCORE_COLUMNS: Sequence[str] = (
    "a59_nature_gravity_duration_score",
    "a60_intentional_negligent_score",
    "a61_mitigate_damage_actions_score",
    "a62_technical_org_measures_score",
    "a63_previous_infringements_score",
    "a64_cooperation_authority_score",
    "a65_data_categories_affected_score",
    "a66_infringement_became_known_score",
    "a67_prior_orders_compliance_score",
    "a68_codes_certification_score",
    "a69_other_factors_score",
)

# Columns included in the slim analysis view used by research tasks.
ANALYSIS_COLUMNS: Sequence[str] = (
    "id",
    "a1_country_code",
    "a2_authority_name",
    "a3_appellate_decision",
    "decision_year",
    "decision_month",
    "decision_date_inferred",
    "temporal_granularity",
    "a8_defendant_class",
    "a9_enterprise_size",
    "a11_defendant_role",
    "a12_sector",
    "a13_sector_other",
    "a15_data_subject_complaint",
    "a16_media_attention",
    "a17_official_audit",
    "a72_cross_border_oss",
    "a73_oss_role",
    "oss_case_category",
    "oss_case_bool",
    "oss_role_lead_bool",
    "oss_role_concerned_bool",
    "fine_imposed_bool",
    "measure_any_bool",
    "measure_count",
    "sanction_profile",
    "fine_present",
    "fine_amount_eur",
    "fine_amount_eur_real_2025",
    "fine_amount_log",
    "fine_fx_method",
    "rights_violated_count",
    "breach_count_total",
    "breach_notification_effect_num",
    "first_violation_status",
    "a70_systematic_art83_discussion",
    "a71_first_violation",
    "art83_discussed_count",
    "art83_aggravating_count",
    "art83_mitigating_count",
    "art83_neutral_count",
    "art83_systematic_bool",
    *SEVERITY_FLAGS,
    *ART83_FACTOR_COLUMNS,
    *ART83_SCORE_COLUMNS,
    *MEASURE_COLUMNS,
    "has_complaint_bool",
    "official_audit_bool",
    "art33_discussed_bool",
    "art33_breached_bool",
)


@dataclass
class LoadResult:
    """Container for the typed dataset and associated diagnostics."""

    data: pd.DataFrame
    diagnostics: Mapping[str, object]


def prepare_output_dir(task_name: str, output_root: Path | None = None) -> Path:
    """Ensure the output directory for a task exists and return it."""

    root = Path(output_root) if output_root is not None else OUTPUT_ROOT
    directory = root / task_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _coerce_boolean(series: pd.Series, fillna: bool = False) -> pd.Series:
    """Convert a series containing boolean-like values to pandas' Boolean dtype."""

    if fillna:
        series = series.where(series.notna(), False)
    converted = series.astype("boolean")
    return converted


def load_typed_enforcement_data(
    data_path: Path | None = None,
    *,
    drop_duplicate_ids: bool = True,
) -> LoadResult:
    """Load the enriched master dataset with harmonised dtypes for analysis."""

    path = Path(data_path) if data_path is not None else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Enriched dataset not found at {path}. Run the enrichment pipeline first."
        )

    raw = pd.read_csv(path)
    original_rows = len(raw)

    duplicate_ids: list[str] = []
    if drop_duplicate_ids:
        dup_mask = raw.duplicated("id", keep=False)
        if dup_mask.any():
            duplicate_ids = sorted(raw.loc[dup_mask, "id"].unique())
        raw = raw.drop_duplicates("id", keep="first")

    # Select relevant columns and coerce dtypes.
    missing_columns = [col for col in ANALYSIS_COLUMNS if col not in raw.columns]
    if missing_columns:
        raise KeyError(
            "The enriched dataset is missing expected columns: "
            + ", ".join(missing_columns)
        )

    data = raw.loc[:, ANALYSIS_COLUMNS].copy()

    # Core identifiers and categorical descriptors.
    data["id"] = data["id"].astype("string")
    data["a1_country_code"] = data["a1_country_code"].astype("category")
    data["a2_authority_name"] = data["a2_authority_name"].astype("string")
    data["a3_appellate_decision"] = data["a3_appellate_decision"].astype("category")
    data["a8_defendant_class"] = data["a8_defendant_class"].astype("category")
    data["a9_enterprise_size"] = data["a9_enterprise_size"].astype("category")
    data["a11_defendant_role"] = data["a11_defendant_role"].astype("category")
    data["a12_sector"] = data["a12_sector"].astype("category")
    data["a13_sector_other"] = data["a13_sector_other"].astype("string")
    data["a15_data_subject_complaint"] = data["a15_data_subject_complaint"].astype(
        "category"
    )
    data["a16_media_attention"] = data["a16_media_attention"].astype("category")
    data["a17_official_audit"] = data["a17_official_audit"].astype("category")
    data["a72_cross_border_oss"] = data["a72_cross_border_oss"].astype("category")
    data["a73_oss_role"] = data["a73_oss_role"].astype("category")
    data["oss_case_category"] = data["oss_case_category"].astype("category")
    data["sanction_profile"] = data["sanction_profile"].astype("category")
    data["a70_systematic_art83_discussion"] = data[
        "a70_systematic_art83_discussion"
    ].astype("category")
    data["a71_first_violation"] = data["a71_first_violation"].astype("category")
    data["first_violation_status"] = data["first_violation_status"].astype("category")

    # Temporal columns.
    data["decision_year"] = pd.to_numeric(
        data["decision_year"], errors="coerce"
    ).astype("Int64")
    data["decision_month"] = pd.to_numeric(
        data["decision_month"], errors="coerce"
    ).astype("Int64")
    data["decision_date_inferred"] = pd.to_datetime(
        data["decision_date_inferred"], errors="coerce"
    )

    # Monetary columns.
    data["fine_amount_eur"] = pd.to_numeric(
        data["fine_amount_eur"], errors="coerce"
    )
    data["fine_amount_eur_real_2025"] = pd.to_numeric(
        data["fine_amount_eur_real_2025"], errors="coerce"
    )
    data["fine_amount_log"] = pd.to_numeric(
        data["fine_amount_log"], errors="coerce"
    )
    data["fine_present"] = _coerce_boolean(data["fine_present"], fillna=True)

    # Boolean conversions for sanction and OSS helpers.
    for column in BOOLEAN_COLUMNS:
        if column not in data:
            continue
        fill_false = column in {"fine_imposed_bool", "measure_any_bool", "fine_present"}
        data[column] = _coerce_boolean(data[column], fillna=fill_false)

    for column in MEASURE_COLUMNS:
        data[column] = _coerce_boolean(data[column], fillna=True)

    # Severity flags and Art. 83 metadata.
    for column in SEVERITY_FLAGS:
        data[column] = data[column].astype("boolean")

    data["rights_violated_count"] = pd.to_numeric(
        data["rights_violated_count"], errors="coerce"
    ).astype("Int64")
    data["breach_count_total"] = pd.to_numeric(
        data["breach_count_total"], errors="coerce"
    ).astype("Int64")
    data["breach_notification_effect_num"] = pd.to_numeric(
        data["breach_notification_effect_num"], errors="coerce"
    )
    data["measure_count"] = pd.to_numeric(
        data["measure_count"], errors="coerce"
    ).fillna(0).astype("Int64")

    data["art83_discussed_count"] = pd.to_numeric(
        data["art83_discussed_count"], errors="coerce"
    ).astype("Int64")
    data["art83_aggravating_count"] = pd.to_numeric(
        data["art83_aggravating_count"], errors="coerce"
    ).astype("Int64")
    data["art83_mitigating_count"] = pd.to_numeric(
        data["art83_mitigating_count"], errors="coerce"
    ).astype("Int64")
    data["art83_neutral_count"] = pd.to_numeric(
        data["art83_neutral_count"], errors="coerce"
    ).astype("Int64")

    data["art83_systematic_bool"] = _coerce_boolean(
        data["art83_systematic_bool"], fillna=False
    )

    for column in ART83_FACTOR_COLUMNS:
        data[column] = data[column].astype("category")
    for column in ART83_SCORE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Provide an explicit fine-in-2025EUR alias expected by the research plan.
    data["fine_eur_2025"] = data["fine_amount_eur_real_2025"]

    diagnostics = {
        "data_path": str(path),
        "original_rows": original_rows,
        "deduplicated_rows": len(data),
        "duplicate_ids": duplicate_ids,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return LoadResult(data=data, diagnostics=diagnostics)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write a JSON payload with UTF-8 encoding."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_summary(
    directory: Path,
    lines: Sequence[str],
    *,
    filename: str = "summary.txt",
    max_lines: int = 10,
) -> Path:
    """Persist a short summary text file (≤ max_lines)."""

    trimmed = list(lines)[:max_lines]
    path = directory / filename
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(trimmed).strip() + "\n")
    return path


def write_memo(
    directory: Path,
    lines: Sequence[str],
    *,
    filename: str = "memo.txt",
) -> Path:
    """Persist a short memo (5–10 lines as per project spec)."""

    content = list(lines)
    path = directory / filename
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(content).strip() + "\n")
    return path


def write_session_info(
    directory: Path,
    extra_packages: Sequence[str] | None = None,
) -> Path:
    """Record package versions used for the current analysis run."""

    packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "statsmodels",
    ]
    if extra_packages:
        packages.extend(extra_packages)

    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue

    path = directory / "session_info.txt"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packages": versions,
    }
    write_json(path, payload)
    return path


def format_bullet_summary(items: Mapping[str, object]) -> str:
    """Create a human-readable bullet summary from a mapping."""

    lines = [f"• {key}: {value}" for key, value in items.items()]
    return "\n".join(lines)


def indent_lines(text: str, spaces: int = 2) -> str:
    """Indent multi-line text for console display."""

    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())
