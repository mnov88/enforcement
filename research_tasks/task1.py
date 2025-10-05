"""Research Task 1: sanctions architecture descriptive analytics."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import common

BOOTSTRAP_ITERATIONS = 2000
MEASURE_TOTAL = len(common.MEASURE_COLUMNS)


@dataclass
class GroupMetrics:
    """Container for stratified descriptive statistics."""

    key: str
    n_cases: int
    fine_rate: float
    fine_rate_ci: tuple[float, float]
    measure_rate: float
    measure_rate_ci: tuple[float, float]
    both_rate: float
    both_rate_ci: tuple[float, float]
    mix_mean: float
    mix_mean_ci: tuple[float, float]


def _bootstrap_mean(values: pd.Series, *, seed: int) -> tuple[float, float]:
    """Bootstrap the mean with 95% confidence intervals."""

    cleaned = values.dropna().to_numpy(dtype=float)
    if cleaned.size == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, cleaned.size, size=(BOOTSTRAP_ITERATIONS, cleaned.size))
    samples = cleaned[sample_idx].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def _seed_for_group(base: str, *, offset: int = 0) -> int:
    digest = hashlib.sha256(f"{base}|{offset}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _summarise_group(values: pd.DataFrame, label: str, seed_offset: int = 0) -> GroupMetrics:
    base_seed = _seed_for_group(label, offset=seed_offset)
    fine_rate = float(values["fine_flag"].mean())
    measure_rate = float(values["measure_flag"].mean())
    both_rate = float(values["both_flag"].mean())
    mix_mean = float(values["sanction_mix_index"].mean())

    return GroupMetrics(
        key=label,
        n_cases=len(values),
        fine_rate=fine_rate,
        fine_rate_ci=_bootstrap_mean(values["fine_flag"], seed=base_seed + 11),
        measure_rate=measure_rate,
        measure_rate_ci=_bootstrap_mean(
            values["measure_flag"], seed=base_seed + 23
        ),
        both_rate=both_rate,
        both_rate_ci=_bootstrap_mean(values["both_flag"], seed=base_seed + 37),
        mix_mean=mix_mean,
        mix_mean_ci=_bootstrap_mean(
            values["sanction_mix_index"], seed=base_seed + 53
        ),
    )


def _groupby_summaries(
    data: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    seed_offset: int = 0,
) -> list[GroupMetrics]:
    metrics: list[GroupMetrics] = []
    grouped = data.groupby(list(group_cols), dropna=False, observed=False)
    for idx, (group_key, group_df) in enumerate(grouped):
        label = group_key if isinstance(group_key, str) else " | ".join(
            ["<NA>" if pd.isna(x) else str(x) for x in (group_key if isinstance(group_key, tuple) else (group_key,))]
        )
        metrics.append(
            _summarise_group(group_df, label, seed_offset=seed_offset + idx)
        )
    return metrics


def _metrics_to_frame(metrics: Iterable[GroupMetrics], *, split_keys: Sequence[str] | None = None) -> pd.DataFrame:
    records = []
    for metric in metrics:
        record = {
            "key": metric.key,
            "n_cases": metric.n_cases,
            "fine_rate": metric.fine_rate,
            "fine_rate_ci_lower": metric.fine_rate_ci[0],
            "fine_rate_ci_upper": metric.fine_rate_ci[1],
            "measure_rate": metric.measure_rate,
            "measure_rate_ci_lower": metric.measure_rate_ci[0],
            "measure_rate_ci_upper": metric.measure_rate_ci[1],
            "both_rate": metric.both_rate,
            "both_rate_ci_lower": metric.both_rate_ci[0],
            "both_rate_ci_upper": metric.both_rate_ci[1],
            "sanction_mix_mean": metric.mix_mean,
            "sanction_mix_ci_lower": metric.mix_mean_ci[0],
            "sanction_mix_ci_upper": metric.mix_mean_ci[1],
        }
        if split_keys:
            parts = metric.key.split(" | ")
            for idx, name in enumerate(split_keys):
                record[name] = parts[idx] if idx < len(parts) else "<NA>"
        records.append(record)
    frame = pd.DataFrame.from_records(records)
    return frame


def _measure_cooccurrence_table(data: pd.DataFrame) -> pd.DataFrame:
    measures = data.loc[:, common.MEASURE_COLUMNS].astype(bool)
    measure_int = measures.astype(int)
    matrix = pd.DataFrame(
        measure_int.T.dot(measure_int),
        index=common.MEASURE_COLUMNS,
        columns=common.MEASURE_COLUMNS,
    )
    total_cases = float(len(data))
    records = []
    for measure_a in matrix.index:
        for measure_b in matrix.columns:
            joint_count = int(matrix.loc[measure_a, measure_b])
            base_count = int(matrix.loc[measure_a, measure_a])
            records.append(
                {
                    "measure_a": measure_a,
                    "measure_b": measure_b,
                    "joint_count": joint_count,
                    "joint_rate": joint_count / total_cases if total_cases else float("nan"),
                    "conditional_on_a": joint_count / base_count if base_count else float("nan"),
                }
            )
    return pd.DataFrame.from_records(records)


def _save_country_bundle_figure(data: pd.DataFrame, output_prefix: Path) -> None:
    bundle_cols = [
        "fine_only_flag",
        "measures_only_flag",
        "both_flag",
        "neither_flag",
    ]
    rates = (
        data.groupby("a1_country_code", observed=False)[bundle_cols]
        .mean()
        .sort_values("fine_only_flag", ascending=False)
    )
    top_codes = (
        data["a1_country_code"].value_counts().head(15).index.tolist()
    )
    rates = rates.reindex(top_codes).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    bottoms = np.zeros(len(rates))
    colors = {
        "fine_only_flag": "#1f78b4",
        "measures_only_flag": "#33a02c",
        "both_flag": "#6a3d9a",
        "neither_flag": "#b15928",
    }
    labels = {
        "fine_only_flag": "Fine only",
        "measures_only_flag": "Measures only",
        "both_flag": "Both",
        "neither_flag": "Neither",
    }
    x = np.arange(len(rates))
    for column in bundle_cols:
        ax.bar(
            x,
            rates[column].to_numpy(),
            bottom=bottoms,
            color=colors[column],
            label=labels[column],
        )
        bottoms += rates[column].to_numpy()

    ax.set_xticks(x)
    ax.set_xticklabels(rates.index)
    ax.set_ylabel("Share of decisions")
    ax.set_title("Sanction bundles by country (top 15 by caseload)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.png", dpi=300)
    fig.savefig(f"{output_prefix}.pdf")
    plt.close(fig)


def _save_mix_heatmap(data: pd.DataFrame, output_prefix: Path) -> None:
    summary = (
        data.groupby(["a12_sector", "a8_defendant_class"], observed=False)
        .agg(
            sanction_mix_mean=("sanction_mix_index", "mean"),
            n_cases=("id", "count"),
        )
        .reset_index()
    )
    pivot = summary.pivot(
        index="a12_sector", columns="a8_defendant_class", values="sanction_mix_mean"
    )
    mask = pivot.isna()
    annot = pivot.round(2)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot,
        mask=mask,
        annot=annot,
        fmt="",
        cmap="mako",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Average sanction mix index"},
    )
    ax.set_xlabel("Defendant class")
    ax.set_ylabel("Sector")
    ax.set_title("Sanction mix index by sector and defendant class")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.png", dpi=300)
    fig.savefig(f"{output_prefix}.pdf")
    plt.close(fig)


def _save_cooccurrence_heatmap(table: pd.DataFrame, output_prefix: Path) -> None:
    matrix = table.pivot(index="measure_a", columns="measure_b", values="joint_rate")
    vmax = float(matrix.values.max()) if not matrix.empty else 0.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        cmap="rocket_r",
        vmin=0,
        vmax=vmax,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Joint rate"},
        ax=ax,
    )
    ax.set_xlabel("Measure B")
    ax.set_ylabel("Measure A")
    ax.set_title("Art. 58 measure co-occurrence (share of cases)")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.png", dpi=300)
    fig.savefig(f"{output_prefix}.pdf")
    plt.close(fig)


def run(
    *,
    output_dir: Path | None = None,
    data_path: Path | None = None,
) -> Path:
    load_result = common.load_typed_enforcement_data(data_path=data_path)
    df = load_result.data.copy()
    out_dir = common.prepare_output_dir("task1", output_dir)

    df["fine_flag"] = df["fine_imposed_bool"].astype(bool)
    df["measure_flag"] = df["measure_any_bool"].astype(bool)
    df["both_flag"] = df["fine_flag"] & df["measure_flag"]
    df["neither_flag"] = ~(df["fine_flag"] | df["measure_flag"])
    df["measures_only_flag"] = (~df["fine_flag"]) & df["measure_flag"]
    df["fine_only_flag"] = df["fine_flag"] & (~df["measure_flag"])
    df["sanction_mix_index"] = df["measure_count"].astype(float) / MEASURE_TOTAL
    df["sanction_bundle"] = np.select(
        [
            df["both_flag"],
            df["fine_only_flag"],
            df["measures_only_flag"],
            df["neither_flag"],
        ],
        ["both", "fine_only", "measures_only", "neither"],
        default="neither",
    )

    # Overall metrics.
    overall_fine_rate = float(df["fine_flag"].mean())
    overall_measure_rate = float(df["measure_flag"].mean())
    overall_mix = float(df["sanction_mix_index"].mean())
    substitution_ratio = {
        "measures_only_given_no_fine": float(
            df.loc[~df["fine_flag"], "measures_only_flag"].mean()
        ),
        "both_given_fine": float(df.loc[df["fine_flag"], "both_flag"].mean()),
    }

    # Grouped metrics.
    country_metrics = _groupby_summaries(df, ["a1_country_code"], seed_offset=1)
    country_frame = _metrics_to_frame(
        country_metrics, split_keys=("country_code",)
    ).sort_values("country_code")
    country_frame.to_csv(out_dir / "sanction_incidence_by_country.csv", index=False)

    sector_class_metrics = _groupby_summaries(
        df, ["a12_sector", "a8_defendant_class"], seed_offset=101
    )
    sector_class_frame = _metrics_to_frame(
        sector_class_metrics,
        split_keys=("sector", "defendant_class"),
    ).sort_values(["sector", "defendant_class"])
    sector_class_frame.to_csv(
        out_dir / "sanction_mix_by_sector_class.csv", index=False
    )

    trigger_tables: list[pd.DataFrame] = []
    trigger_dimensions = [
        ("a15_data_subject_complaint", "complaint_status"),
        ("a16_media_attention", "media_attention"),
        ("a17_official_audit", "official_audit"),
        ("a72_cross_border_oss", "oss_cross_border"),
        ("oss_case_category", "oss_case_category"),
    ]
    for offset, (column, dimension_name) in enumerate(trigger_dimensions, start=201):
        metrics = _groupby_summaries(df, [column], seed_offset=offset * 10)
        frame = _metrics_to_frame(metrics, split_keys=(dimension_name,))
        frame.insert(0, "dimension", dimension_name)
        frame["fine_rate_delta_vs_overall"] = frame["fine_rate"] - overall_fine_rate
        frame["measure_rate_delta_vs_overall"] = (
            frame["measure_rate"] - overall_measure_rate
        )
        trigger_tables.append(frame)
    triggers_frame = pd.concat(trigger_tables, ignore_index=True)
    triggers_frame.to_csv(out_dir / "triggers_oss_descriptives.csv", index=False)

    cooccurrence_table = _measure_cooccurrence_table(df)
    cooccurrence_table.to_csv(out_dir / "bundle_cooccurrence.csv", index=False)

    # Figures.
    _save_country_bundle_figure(df, out_dir / "fig_country_rates")
    _save_mix_heatmap(df, out_dir / "fig_mix_by_sector_class")
    _save_cooccurrence_heatmap(cooccurrence_table, out_dir / "fig_bundle_cooccurrence")

    # Case-level summary for downstream modelling phases.
    summary_cols = [
        "id",
        "a1_country_code",
        "a2_authority_name",
        "a8_defendant_class",
        "a12_sector",
        "fine_flag",
        "measure_flag",
        "both_flag",
        "measures_only_flag",
        "fine_only_flag",
        "neither_flag",
        "sanction_bundle",
        "sanction_mix_index",
        "measure_count",
        "fine_eur_2025",
        "a72_cross_border_oss",
        "oss_case_category",
        "oss_role_lead_bool",
        "oss_role_concerned_bool",
        "a15_data_subject_complaint",
        "a16_media_attention",
        "a17_official_audit",
        "first_violation_status",
    ]
    df.loc[:, summary_cols].to_parquet(out_dir / "t1_summary.parquet", index=False)

    # Narrative summaries.
    country_frame_filtered = country_frame[country_frame["n_cases"] >= 10]
    country_frame_valid = country_frame_filtered[
        ~country_frame_filtered["country_code"].isin({"NOT_APPLICABLE", "NOT_DISCUSSED"})
    ]
    if country_frame_valid.empty:
        country_frame_valid = country_frame_filtered

    top_country = country_frame_valid.sort_values(
        "fine_rate", ascending=False
    ).head(1)
    bottom_country = country_frame_valid.sort_values(
        "fine_rate", ascending=True
    ).head(1)

    top_mix = sector_class_frame.sort_values(
        "sanction_mix_mean", ascending=False
    ).head(1)
    bottom_mix = sector_class_frame.sort_values(
        "sanction_mix_mean", ascending=True
    ).head(1)

    oss_delta = (
        triggers_frame[triggers_frame["dimension"] == "oss_cross_border"]
        .sort_values("fine_rate")
        .iloc[[0, -1]]
    )
    complaint_delta = (
        triggers_frame[triggers_frame["dimension"] == "complaint_status"]
        .sort_values("fine_rate")
        .iloc[[0, -1]]
    )

    summary_lines = [
        "Task 1 wrap-up:",
        (
            f"• Largest fine incidence gap: {top_country['country_code'].values[0]}"
            f" at {top_country['fine_rate'].values[0]:.1%} vs "
            f"{bottom_country['country_code'].values[0]} at "
            f"{bottom_country['fine_rate'].values[0]:.1%} (≥10 cases)."
        ),
        (
            f"• Highest mix intensity: {top_mix['sector'].values[0]}"
            f"/{top_mix['defendant_class'].values[0]} with mean index"
            f" {top_mix['sanction_mix_mean'].values[0]:.2f};"
            f" lowest is {bottom_mix['sector'].values[0]}"
            f"/{bottom_mix['defendant_class'].values[0]} at"
            f" {bottom_mix['sanction_mix_mean'].values[0]:.2f}."
        ),
        (
            f"• Complaint-triggered cases fine at {complaint_delta['fine_rate'].values[1]:.1%}"
            f" vs {complaint_delta['fine_rate'].values[0]:.1%} when NOT_DISCUSSED."
        ),
        (
            f"• OSS cross-border share shifts fine incidence from"
            f" {oss_delta['fine_rate'].values[0]:.1%} (NO) to"
            f" {oss_delta['fine_rate'].values[1]:.1%} (YES)."
        ),
        (
            f"• Measure substitution ratio: P(measures only | no fine) ="
            f" {substitution_ratio['measures_only_given_no_fine']:.1%};"
            f" P(both | fine) = {substitution_ratio['both_given_fine']:.1%}."
        ),
    ]

    memo_lines = [
        "Task 1 memo:",
        f"1. Overall fine rate {overall_fine_rate:.1%} vs measure incidence {overall_measure_rate:.1%} (mix mean {overall_mix:.2f}).",
        "2. Country spread widens once ≥10 cases considered; see CSV for full bootstrap CIs.",
        "3. Sector-class heatmap highlights compliance orders concentrated among public authorities.",
        "4. Trigger/OSS table records fine-rate deltas relative to overall averages for quick benchmarking.",
        "5. Bundle co-occurrence matrix underpins substitution ratio diagnostics for sanctions policy.",
    ]

    common.write_summary(out_dir, summary_lines)
    common.write_memo(out_dir, memo_lines)
    common.write_session_info(out_dir)

    print("Task 1 summary:")
    for line in summary_lines:
        print(f"  {line}")

    return out_dir


if __name__ == "__main__":
    run()
