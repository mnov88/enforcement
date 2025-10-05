"""Research Task 0: sanity checks and analysis-ready view."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import common

KEY_FIELDS: Sequence[str] = (
    "id",
    "a1_country_code",
    "a2_authority_name",
    "a8_defendant_class",
    "a12_sector",
    "decision_year",
    "decision_date_inferred",
    "fine_imposed_bool",
    "measure_any_bool",
    "fine_eur_2025",
    "measure_count",
    "sanction_profile",
    "a72_cross_border_oss",
    "oss_case_category",
    "oss_role_lead_bool",
    "oss_role_concerned_bool",
    "a15_data_subject_complaint",
    "a16_media_attention",
    "a17_official_audit",
    "breach_has_art5",
    "breach_has_art6",
    "breach_has_art32",
    "rights_violated_count",
    "art83_discussed_count",
    "art83_systematic_bool",
    "first_violation_status",
)


def _missing_heatmap(data: pd.DataFrame, output_path: Path) -> None:
    """Create a missingness heatmap for key analytical fields."""

    heatmap_data = data.loc[:, KEY_FIELDS].isna().astype(float)
    # Restrict to a manageable number of rows for plotting clarity.
    sample = heatmap_data.head(250)

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        sample.T,
        cmap="viridis",
        cbar_kws={"label": "Missing"},
        linewidths=0.2,
        linecolor="white",
    )
    plt.title("Task 0 – Missingness across key governance fields (first 250 cases)")
    plt.xlabel("Case index")
    plt.ylabel("Fields")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _one_pager(
    output_path: Path,
    *,
    summary_lines: Sequence[str],
    missing_top: Sequence[tuple[str, float]],
    duplicate_ids: Sequence[str],
) -> None:
    """Write a PDF one-pager summarising readiness signals."""

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    content_lines = [
        "GDPR Enforcement – Research Task 0",
        f"Generated: {timestamp}",
        "",
        "Readiness snapshot:",
        *[f"• {line}" for line in summary_lines],
        "",
        "Highest missingness (top 6 fields):",
        *[f"• {field}: {value:.1%}" for field, value in missing_top],
    ]
    if duplicate_ids:
        content_lines.extend(
            [
                "",
                "Duplicate case identifiers removed:",
                *[f"• {case_id}" for case_id in duplicate_ids],
            ]
        )

    wrapped_text = "\n".join(content_lines)
    ax.text(
        0.02,
        0.98,
        wrapped_text,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run(
    *,
    output_dir: Path | None = None,
    data_path: Path | None = None,
) -> Path:
    """Execute Research Task 0 and persist artefacts."""

    load_result = common.load_typed_enforcement_data(data_path=data_path)
    data = load_result.data
    diagnostics = load_result.diagnostics

    out_dir = common.prepare_output_dir("task0", output_dir)

    # Persist the typed subset for downstream tasks.
    analysis_path = out_dir / "analysis_view.parquet"
    data.to_parquet(analysis_path, index=False)

    # Data quality summary (missingness) for key analytical fields.
    missing_pct = {
        field: float(data[field].isna().mean()) for field in KEY_FIELDS
    }
    data_check_payload = {
        "row_count": int(len(data)),
        "column_count": int(data.shape[1]),
        "missing_pct": missing_pct,
        "duplicate_ids_removed": diagnostics["duplicate_ids"],
        "source_rows": diagnostics["original_rows"],
        "generated_at": diagnostics["generated_at"],
    }
    common.write_json(out_dir / "data_check.json", data_check_payload)

    # Visual diagnostics.
    _missing_heatmap(data, out_dir / "missingness_heatmap.png")

    # Aggregate stats for summary.
    fine_rate = float(data["fine_imposed_bool"].mean())
    measure_rate = float(data["measure_any_bool"].mean())
    both_rate = float(
        (data["fine_imposed_bool"] & data["measure_any_bool"]).mean()
    )
    mix_index = float(
        (data["measure_count"] / len(common.MEASURE_COLUMNS)).mean()
    )
    oss_share = float(
        data["a72_cross_border_oss"].isin(["YES"]).mean()
    )
    fine_amount_missing = float(data["fine_eur_2025"].isna().mean())

    summary_lines = [
        f"{len(data):,} unique decisions from {data['a1_country_code'].nunique()} DPAs (out of {diagnostics['original_rows']:,} raw rows).",
        f"Fine incidence {fine_rate:.1%}, measures applied {measure_rate:.1%}, combined sanctions {both_rate:.1%}.",
        f"Average sanction mix index {mix_index:.2f} (0=no measures, 1=all eight measures).",
        f"Cross-border OSS share {oss_share:.1%}; fine amount in 2025 EUR missing {fine_amount_missing:.1%}.",
    ]

    missing_top = sorted(missing_pct.items(), key=lambda kv: kv[1], reverse=True)[:6]
    _one_pager(
        out_dir / "data_readiness_one_pager.pdf",
        summary_lines=summary_lines,
        missing_top=missing_top,
        duplicate_ids=diagnostics["duplicate_ids"],
    )

    # Console display and short memo outputs.
    print("Task 0 summary:")
    for line in summary_lines:
        print(f"  - {line}")

    memo_lines = [
        "Task 0 memo:",
        f"1. Typed analysis view saved for downstream research phases ({analysis_path.name}).",
        f"2. Fine incidence registered at {fine_rate:.1%} with measures at {measure_rate:.1%}.",
        f"3. Combined sanctions (fine + measures) appear in {both_rate:.1%} of decisions; mix index averages {mix_index:.2f}.",
        f"4. OSS cross-border cases represent {oss_share:.1%} of the portfolio.",
        f"5. Normalised 2025-euro fines are absent in {fine_amount_missing:.1%} of records, flagging measure-only outcomes.",
    ]

    common.write_summary(out_dir, summary_lines)
    common.write_memo(out_dir, memo_lines)
    common.write_session_info(out_dir)

    return out_dir


if __name__ == "__main__":
    run()
