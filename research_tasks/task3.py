"""Research Task 3 – Harmonization & heterogeneity diagnostics."""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.bayes_mixed_glm import (
    BayesMixedGLMResults,
    BinomialBayesMixedGLM,
)

from . import common

EXTRA_COLUMNS: Sequence[str] = (
    "id",
    "breach_notification_effect_num",
    "has_cookies",
)

NN_K_VALUES: tuple[int, ...] = (1, 2, 3)
MIN_CATEGORY_SIZE = 20
FINE_AMOUNT_COLUMN = "fine_amount_eur_real_2025"


@dataclass
class MixedEffectsResults:
    """Container for two-part mixed-effects outputs."""

    logit_result: BayesMixedGLMResults
    linear_result: sm.regression.mixed_linear_model.MixedLMResults
    logit_variances: dict[str, float]
    linear_variances: dict[str, float]


def _clean_repeat_offender(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    token = str(value).strip().upper()
    if token == "NO_REPEAT_OFFENDER":
        return True
    if token in {"YES_FIRST_TIME", "NOT_DISCUSSED", "NOT_DISCISSED"}:
        return False
    return False


def _prepare_dataset(data: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    merged = data.merge(extra, on="id", how="left")

    merged["fine_flag"] = merged["fine_imposed_bool"].fillna(False).astype(bool)
    merged["measure_flag"] = merged["measure_any_bool"].fillna(False).astype(bool)
    merged["measures_only_flag"] = (~merged["fine_flag"]) & merged["measure_flag"]
    merged["fine_only_flag"] = merged["fine_flag"] & (~merged["measure_flag"])
    merged["both_flag"] = merged["fine_flag"] & merged["measure_flag"]
    merged["neither_flag"] = ~(merged["fine_flag"] | merged["measure_flag"])
    merged["sanction_bundle"] = pd.Categorical(
        np.select(
            [
                merged["both_flag"],
                merged["fine_only_flag"],
                merged["measures_only_flag"],
                merged["neither_flag"],
            ],
            ["both", "fine_only", "measures_only", "neither"],
            default="neither",
        ),
        categories=["neither", "fine_only", "measures_only", "both"],
        ordered=False,
    )

    merged["rights_violated_count"] = pd.to_numeric(
        merged["rights_violated_count"], errors="coerce"
    ).fillna(0)
    merged["breach_count_total"] = pd.to_numeric(
        merged["breach_count_total"], errors="coerce"
    ).fillna(0)
    merged["complaint_flag"] = merged["has_complaint_bool"].fillna(False).astype(bool)
    merged["audit_flag"] = merged["official_audit_bool"].fillna(False).astype(bool)
    merged["media_attention_flag"] = merged["a16_media_attention"].fillna(
        "NOT_DISCUSSED"
    ).eq("YES")
    merged["oss_case_bool"] = merged["oss_case_bool"].fillna(False).astype(bool)
    merged["oss_role_lead_bool"] = merged["oss_role_lead_bool"].fillna(False).astype(bool)
    merged["oss_role_concerned_bool"] = merged["oss_role_concerned_bool"].fillna(False).astype(bool)
    merged["repeat_offender"] = merged["first_violation_status"].map(
        _clean_repeat_offender
    )
    merged["self_report_flag"] = merged["breach_notification_effect_num"].fillna(0).lt(0)
    merged["has_cookies_flag"] = merged["has_cookies"].fillna(False).astype(bool)

    merged[FINE_AMOUNT_COLUMN] = pd.to_numeric(
        merged[FINE_AMOUNT_COLUMN], errors="coerce"
    )
    merged["log_fine_amount"] = np.log1p(merged[FINE_AMOUNT_COLUMN])
    merged = merged[merged["decision_year"].notna()].copy()
    merged["decision_year"] = merged["decision_year"].astype(int)

    for column in [
        "a2_authority_name",
        "a1_country_code",
        "a8_defendant_class",
        "a12_sector",
        "a72_cross_border_oss",
        "a9_enterprise_size",
        "a11_defendant_role",
    ]:
        if column in merged:
            merged[column] = merged[column].astype("category")

    return merged


def _categorical_group(series: pd.Series, minimum: int) -> pd.Series:
    counts = series.value_counts(dropna=False)
    rare = counts[counts < minimum].index
    return series.astype(str).where(~series.isin(rare), "OTHER")


def _build_similarity_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(index=df.index)
    features["rights_violated_count"] = df["rights_violated_count"].astype(float)
    features["breach_count_total"] = df["breach_count_total"].astype(float)
    features["decision_year_centered"] = df["decision_year"] - df["decision_year"].mean()

    bool_columns = [
        "breach_has_art5",
        "breach_has_art6",
        "breach_has_art32",
        "complaint_flag",
        "audit_flag",
        "media_attention_flag",
        "self_report_flag",
        "oss_case_bool",
        "oss_role_lead_bool",
        "oss_role_concerned_bool",
        "repeat_offender",
    ]
    for column in bool_columns:
        if column in df:
            features[column] = df[column].fillna(False).astype(int)
        else:
            features[column] = 0

    sector_group = _categorical_group(df["a12_sector"], MIN_CATEGORY_SIZE)
    class_group = _categorical_group(df["a8_defendant_class"], MIN_CATEGORY_SIZE)
    role_group = _categorical_group(df["a11_defendant_role"], MIN_CATEGORY_SIZE)
    size_group = df["a9_enterprise_size"].astype(str).fillna("UNKNOWN")
    oss_group = df["a72_cross_border_oss"].astype(str).fillna("NOT_DISCUSSED")

    cat_frame = pd.get_dummies(
        pd.DataFrame(
            {
                "sector_group": sector_group,
                "class_group": class_group,
                "role_group": role_group,
                "size_group": size_group,
                "oss_group": oss_group,
            }
        ),
        dummy_na=False,
    )
    features = pd.concat([features, cat_frame], axis=1)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled, columns=features.columns, index=df.index)
    return features, scaled_df


def _nearest_neighbor_pairs(
    df: pd.DataFrame,
    scaled_features: pd.DataFrame,
    k_values: Iterable[int],
) -> pd.DataFrame:
    max_k = max(k_values)
    model = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
    model.fit(scaled_features)
    distances, indices = model.kneighbors(scaled_features)

    records: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    for i, row in enumerate(df.itertuples(index=False)):
        for rank, neighbor_idx in enumerate(indices[i, 1 : max_k + 1], start=1):
            target = df.iloc[int(neighbor_idx)]
            pair_ids = tuple(sorted((row.id, target["id"])))
            key = (pair_ids[0], pair_ids[1], rank)
            if key in seen:
                continue
            seen.add(key)

            pair_type = (
                "cross_country"
                if row.a1_country_code != target["a1_country_code"]
                else "within_country"
            )
            both_fined = bool(row.fine_flag and target["fine_flag"])
            fine_gap = (
                float(row.fine_amount_eur_real_2025 - target[FINE_AMOUNT_COLUMN])
                if both_fined
                else np.nan
            )
            log_gap = (
                float(row.log_fine_amount - target["log_fine_amount"])
                if both_fined
                else np.nan
            )
            records.append(
                {
                    "source_id": row.id,
                    "target_id": target["id"],
                    "source_country": row.a1_country_code,
                    "target_country": target["a1_country_code"],
                    "source_authority": row.a2_authority_name,
                    "target_authority": target["a2_authority_name"],
                    "distance": float(distances[i, rank]),
                    "k": rank,
                    "pair_type": pair_type,
                    "both_fined_flag": both_fined,
                    "fine_gap_eur_abs": abs(fine_gap) if not math.isnan(fine_gap) else np.nan,
                    "fine_gap_log_abs": abs(log_gap) if not math.isnan(log_gap) else np.nan,
                    "bundle_disagreement_flag": row.sanction_bundle != target["sanction_bundle"],
                    "sanction_bundle_source": row.sanction_bundle,
                    "sanction_bundle_target": target["sanction_bundle"],
                    "same_sector_flag": row.a12_sector == target["a12_sector"],
                    "same_class_flag": row.a8_defendant_class == target["a8_defendant_class"],
                }
            )
    return pd.DataFrame.from_records(records)


def _summarise_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for (pair_type, k), group in pairs.groupby(["pair_type", "k"], dropna=False):
        fined = group[group["both_fined_flag"]]
        summaries.append(
            {
                "pair_type": pair_type,
                "k": int(k),
                "n_pairs": int(len(group)),
                "median_distance": float(group["distance"].median()),
                "bundle_disagreement_rate": float(group["bundle_disagreement_flag"].mean()),
                "median_fine_gap_eur": float(fined["fine_gap_eur_abs"].median()) if not fined.empty else np.nan,
                "median_log_gap": float(fined["fine_gap_log_abs"].median()) if not fined.empty else np.nan,
            }
        )
    return pd.DataFrame.from_records(summaries)


def _plot_nn_gap_distribution(pairs: pd.DataFrame, output_prefix: Path) -> None:
    subset = pairs[(pairs["k"] == 1) & (pairs["both_fined_flag"])]
    if subset.empty:
        return
    plt.figure(figsize=(8, 5))
    ax = sns.violinplot(
        data=subset,
        x="pair_type",
        y="fine_gap_eur_abs",
        inner="box",
        cut=0,
        hue="pair_type",
        palette="Set2",
        dodge=False,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    plt.ylabel("Absolute fine gap (EUR, 2025)")
    plt.xlabel("Pair type")
    plt.title("Nearest-neighbour fine gaps (k=1, fined pairs)")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300)
    plt.savefig(f"{output_prefix}.pdf")
    plt.close()


def _fit_logit_mixed(
    df: pd.DataFrame,
) -> tuple[BinomialBayesMixedGLMResults, dict[str, float]]:
    frame = df.copy()
    for column in [
        "fine_flag",
        "breach_has_art32",
        "complaint_flag",
        "audit_flag",
        "media_attention_flag",
        "self_report_flag",
        "oss_case_bool",
        "repeat_offender",
    ]:
        frame[column] = frame[column].astype(int)

    logit_formula = (
        "fine_flag ~ rights_violated_count + breach_has_art32 + complaint_flag + "
        "audit_flag + media_attention_flag + self_report_flag + oss_case_bool + "
        "repeat_offender + C(a8_defendant_class) + C(a12_sector) + "
        "C(a72_cross_border_oss) + C(decision_year)"
    )
    vc = {
        "authority": "0 + C(a2_authority_name)",
        "country": "0 + C(a1_country_code)",
    }
    model = BinomialBayesMixedGLM.from_formula(logit_formula, vc, data=frame)
    result = model.fit_vb()
    log_sd_authority, log_sd_country = result.vcp_mean
    variance_authority = float(np.exp(log_sd_authority) ** 2)
    variance_country = float(np.exp(log_sd_country) ** 2)
    logistic_residual = math.pi**2 / 3
    return result, {
        "authority": variance_authority,
        "country": variance_country,
        "residual": logistic_residual,
    }


def _fit_linear_mixed(df: pd.DataFrame) -> tuple[sm.regression.mixed_linear_model.MixedLMResults, dict[str, float]]:
    fined = df[df["fine_flag"] & df["log_fine_amount"].notna()].copy()
    if fined.empty:
        raise ValueError("No fined observations available for linear mixed model.")

    for column in [
        "breach_has_art32",
        "complaint_flag",
        "audit_flag",
        "media_attention_flag",
        "self_report_flag",
        "oss_case_bool",
        "repeat_offender",
    ]:
        fined[column] = fined[column].astype(int)

    fined["sector_group"] = _categorical_group(
        fined["a12_sector"], MIN_CATEGORY_SIZE
    )
    fined["class_group"] = _categorical_group(
        fined["a8_defendant_class"], MIN_CATEGORY_SIZE
    )
    fined["decision_year_centered"] = fined["decision_year"] - fined["decision_year"].mean()

    for column in ["sector_group", "class_group", "a2_authority_name", "a1_country_code"]:
        fined[column] = fined[column].astype("category")

    formula = (
        "log_fine_amount ~ rights_violated_count + breach_has_art32 + complaint_flag + "
        "audit_flag + media_attention_flag + self_report_flag + oss_case_bool + "
        "repeat_offender + C(class_group) + C(sector_group) + decision_year_centered"
    )
    mixed_model = sm.MixedLM.from_formula(
        formula,
        groups="a2_authority_name",
        re_formula="1",
        vc_formula={"country": "0 + C(a1_country_code)"},
        data=fined,
    )
    result = mixed_model.fit(reml=False, method="lbfgs")
    variance_authority = float(result.cov_re.iloc[0, 0])
    variance_country = float(result.vcomp[0]) if result.vcomp.size else 0.0
    residual_variance = float(result.scale)
    return result, {
        "authority": variance_authority,
        "country": variance_country,
        "residual": residual_variance,
    }


def _variance_components_table(
    logit_variances: dict[str, float],
    linear_variances: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for model_label, components in (
        ("incidence", logit_variances),
        ("magnitude", linear_variances),
    ):
        total = sum(components.values())
        for component, variance in components.items():
            share = variance / total if total else np.nan
            rows.append(
                {
                    "model_part": model_label,
                    "component": component,
                    "variance": variance,
                    "share": share,
                }
            )
    return pd.DataFrame.from_records(rows)


def _plot_variance_components(table: pd.DataFrame, output_prefix: Path) -> None:
    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=table,
        x="model_part",
        y="share",
        hue="component",
        palette="viridis",
    )
    plt.ylabel("Share of total variance")
    plt.xlabel("Model part")
    plt.ylim(0, 1)
    plt.legend(title="Component", frameon=False)
    plt.title("Random-effect variance shares")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300)
    plt.savefig(f"{output_prefix}.pdf")
    plt.close()


def _interaction_tables(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    sector_class = (
        df.groupby(["a12_sector", "a8_defendant_class"], observed=False)
        .agg(
            n_cases=("id", "size"),
            fine_rate=("fine_flag", "mean"),
            median_fine_eur=(
                FINE_AMOUNT_COLUMN,
                lambda s: float(s.dropna().median()),
            ),
        )
        .reset_index()
    )
    sector_class.insert(0, "interaction", "sector_x_class")
    records.extend(sector_class.to_dict("records"))

    cookies = df[df["has_cookies_flag"]]
    if not cookies.empty:
        cookies_group = (
            cookies.groupby("a1_country_code", observed=False)
            .agg(
                n_cases=("id", "size"),
                fine_rate=("fine_flag", "mean"),
                median_fine_eur=(
                    FINE_AMOUNT_COLUMN,
                    lambda s: float(s.dropna().median()),
                ),
            )
            .reset_index()
        )
        cookies_group.insert(0, "interaction", "cookies_by_country")
        cookies_group.rename(columns={"a1_country_code": "level_1"}, inplace=True)
        records.extend(cookies_group.to_dict("records"))

    telecom = df[df["a12_sector"].astype(str) == "TELECOM"]
    if not telecom.empty:
        telecom_group = (
            telecom.groupby("a1_country_code", observed=False)
            .agg(
                n_cases=("id", "size"),
                fine_rate=("fine_flag", "mean"),
                median_fine_eur=(
                    FINE_AMOUNT_COLUMN,
                    lambda s: float(s.dropna().median()),
                ),
            )
            .reset_index()
        )
        telecom_group.insert(0, "interaction", "telecom_by_country")
        telecom_group.rename(columns={"a1_country_code": "level_1"}, inplace=True)
        records.extend(telecom_group.to_dict("records"))

    trigger_map = {
        "complaint_flag": "complaint",
        "audit_flag": "official_audit",
        "media_attention_flag": "media_attention",
        "self_report_flag": "self_report",
    }
    for column, label in trigger_map.items():
        trig_df = df[df[column].astype(bool)]
        if trig_df.empty:
            continue
        group = (
            trig_df.groupby("oss_case_bool", observed=False)
            .agg(
                n_cases=("id", "size"),
                fine_rate=("fine_flag", "mean"),
                median_fine_eur=(
                    FINE_AMOUNT_COLUMN,
                    lambda s: float(s.dropna().median()),
                ),
            )
            .reset_index()
        )
        group.insert(0, "interaction", "trigger_x_oss")
        group.insert(1, "level_1", label)
        group.rename(columns={"oss_case_bool": "level_2"}, inplace=True)
        records.extend(group.to_dict("records"))

    return pd.DataFrame.from_records(records)


def _plot_interactions(
    df: pd.DataFrame,
    interactions: pd.DataFrame,
    output_prefix: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    grid = plt.GridSpec(2, 2, figure=plt.gcf())

    pivot = (
        df.pivot_table(
            values="fine_flag",
            index="a12_sector",
            columns="a8_defendant_class",
            aggfunc="mean",
            observed=False,
        )
        .fillna(0)
    )
    ax1 = plt.subplot(grid[:, 0])
    sns.heatmap(pivot, ax=ax1, cmap="magma", cbar_kws={"label": "Fine rate"})
    ax1.set_title("Fine incidence by sector × class")

    cookies = interactions[interactions["interaction"] == "cookies_by_country"]
    telecom = interactions[interactions["interaction"] == "telecom_by_country"]
    ax2 = plt.subplot(grid[0, 1])
    if not cookies.empty:
        top = cookies.sort_values("n_cases", ascending=False).head(10)
        sns.barplot(
            data=top,
            x="level_1",
            y="fine_rate",
            ax=ax2,
            color="#1f78b4",
        )
        ax2.set_ylabel("Fine rate")
        ax2.set_xlabel("Country")
        ax2.set_title("Cookies cases – top countries")
    else:
        ax2.set_axis_off()

    ax3 = plt.subplot(grid[1, 1])
    if not telecom.empty:
        top = telecom.sort_values("n_cases", ascending=False).head(10)
        sns.barplot(
            data=top,
            x="level_1",
            y="fine_rate",
            ax=ax3,
            color="#33a02c",
        )
        ax3.set_ylabel("Fine rate")
        ax3.set_xlabel("Country")
        ax3.set_title("Telecom cases – top countries")
    else:
        ax3.set_axis_off()

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300)
    plt.savefig(f"{output_prefix}.pdf")
    plt.close()


def _within_authority_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    focus = df[df["a8_defendant_class"].isin(["PUBLIC", "PRIVATE"])]
    if focus.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    grouped = focus.groupby("a2_authority_name", observed=False)
    for authority, group in grouped:
        public = group[group["a8_defendant_class"] == "PUBLIC"]
        private = group[group["a8_defendant_class"] == "PRIVATE"]
        if public.empty or private.empty:
            continue
        records.append(
            {
                "authority": authority,
                "n_public": int(len(public)),
                "n_private": int(len(private)),
                "fine_rate_public": float(public["fine_flag"].mean()),
                "fine_rate_private": float(private["fine_flag"].mean()),
                "fine_rate_gap": float(private["fine_flag"].mean() - public["fine_flag"].mean()),
                "median_fine_public": float(
                    public[FINE_AMOUNT_COLUMN].dropna().median()
                ),
                "median_fine_private": float(
                    private[FINE_AMOUNT_COLUMN].dropna().median()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _plot_within_authority(contrasts: pd.DataFrame, output_prefix: Path) -> None:
    if contrasts.empty:
        return
    top = contrasts.sort_values("fine_rate_gap", ascending=False).head(12)
    plt.figure(figsize=(10, 5))
    plot_data = top.assign(
        gap_label=np.where(
            top["fine_rate_gap"] >= 0, "Private higher", "Public higher"
        )
    )
    sns.barplot(
        data=plot_data,
        x="fine_rate_gap",
        y="authority",
        hue="gap_label",
        orient="h",
        palette={"Private higher": "#b2182b", "Public higher": "#2166ac"},
    )
    plt.legend(title="Fine-rate gap", frameon=False)
    plt.axvline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Private minus public fine rate")
    plt.ylabel("Authority")
    plt.title("Within-authority fine-rate gaps")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300)
    plt.savefig(f"{output_prefix}.pdf")
    plt.close()


def run(
    *,
    output_dir: Path | None = None,
    data_path: Path | None = None,
) -> Path:
    load_result = common.load_typed_enforcement_data(data_path=data_path)
    extra = pd.read_csv(
        data_path if data_path is not None else common.DATA_PATH,
        usecols=list(EXTRA_COLUMNS),
    )
    dataset = _prepare_dataset(load_result.data, extra)

    out_dir = common.prepare_output_dir("task3", output_dir)

    feature_frame, scaled_features = _build_similarity_features(dataset)
    pairs = _nearest_neighbor_pairs(dataset, scaled_features, NN_K_VALUES)
    pair_summary = _summarise_pairs(pairs)

    pairs.to_csv(out_dir / "nn_pairs.csv", index=False)
    pair_summary.to_csv(out_dir / "nn_gap_summary.csv", index=False)
    feature_export = feature_frame.copy()
    feature_export.insert(0, "country", dataset["a1_country_code"].astype(str))
    feature_export.insert(0, "id", dataset["id"].astype(str))
    feature_export.reset_index(drop=True).to_feather(out_dir / "t3_nn_index.feather")

    _plot_nn_gap_distribution(pairs, out_dir / "fig_nn_gap_distribution")

    logit_result, logit_variances = _fit_logit_mixed(dataset)
    linear_result, linear_variances = _fit_linear_mixed(dataset)
    variance_table = _variance_components_table(logit_variances, linear_variances)
    variance_table.to_csv(
        out_dir / "mixed_effects_variance_components.csv", index=False
    )
    _plot_variance_components(variance_table, out_dir / "fig_variance_components")

    interactions = _interaction_tables(dataset)
    interactions.to_csv(out_dir / "heterogeneity_interactions.csv", index=False)
    _plot_interactions(dataset, interactions, out_dir / "fig_interaction_effects")

    contrasts = _within_authority_contrasts(dataset)
    contrasts.to_csv(out_dir / "within_authority_contrasts.csv", index=False)
    _plot_within_authority(contrasts, out_dir / "fig_within_authority")

    with (out_dir / "t3_mixed_effects_models.pkl").open("wb") as handle:
        pickle.dump(
            MixedEffectsResults(
                logit_result=logit_result,
                linear_result=linear_result,
                logit_variances=logit_variances,
                linear_variances=linear_variances,
            ),
            handle,
        )

    cross_k1 = pair_summary[
        (pair_summary["pair_type"] == "cross_country") & (pair_summary["k"] == 1)
    ]
    cross_gap = (
        float(cross_k1["median_fine_gap_eur"].dropna().iloc[0])
        if not cross_k1.empty and cross_k1["median_fine_gap_eur"].notna().any()
        else None
    )
    overall_disagreement = (
        float(
            pair_summary[pair_summary["k"] == 1][
                "bundle_disagreement_rate"
            ].mean()
        )
        if not pair_summary.empty
        else None
    )
    incidence_authority_share = (
        variance_table[
            (variance_table["model_part"] == "incidence")
            & (variance_table["component"] == "authority")
        ]["share"].iloc[0]
        if not variance_table.empty
        and not variance_table[
            (variance_table["model_part"] == "incidence")
            & (variance_table["component"] == "authority")
        ]["share"].empty
        else None
    )
    incidence_country_share = (
        variance_table[
            (variance_table["model_part"] == "incidence")
            & (variance_table["component"] == "country")
        ]["share"].iloc[0]
        if not variance_table.empty
        and not variance_table[
            (variance_table["model_part"] == "incidence")
            & (variance_table["component"] == "country")
        ]["share"].empty
        else None
    )
    magnitude_authority_share = (
        variance_table[
            (variance_table["model_part"] == "magnitude")
            & (variance_table["component"] == "authority")
        ]["share"].iloc[0]
        if not variance_table.empty
        and not variance_table[
            (variance_table["model_part"] == "magnitude")
            & (variance_table["component"] == "authority")
        ]["share"].empty
        else None
    )

    summary_lines = [
        "Task 3 wrap-up:",
        (
            f"• Median cross-country fine gap (k=1) at {cross_gap:.0f} EUR"
            if cross_gap is not None
            else "• Cross-country fine gaps reported in nn_gap_summary.csv."
        ),
        (
            f"• Bundle disagreement for nearest neighbours: {overall_disagreement:.1%}."
            if overall_disagreement is not None
            else "• Bundle disagreement rates available in summary table."
        ),
        (
            f"• Random-intercept share (incidence – authority/country): {incidence_authority_share:.1%} / {incidence_country_share:.1%}."
            if incidence_authority_share is not None and incidence_country_share is not None
            else "• Variance components recorded in CSV."
        ),
        (
            f"• Magnitude model assigns {magnitude_authority_share:.1%} variance to authorities."
            if magnitude_authority_share is not None
            else "• Magnitude variance decomposition available in CSV."
        ),
        (
            f"• {len(contrasts)} authorities exhibit both public & private defendants; gaps saved to within_authority_contrasts.csv."
        ),
    ]

    memo_lines = [
        "Task 3 memo:",
        "1. Nearest-neighbour analysis benchmarks cross-country vs domestic gaps (k=1–3).",
        "2. Mixed-effects two-part model quantifies authority and country variance shares.",
        "3. Interaction tables flag sector×class contrasts plus cookies/telecom country deltas.",
        "4. Trigger×OSS table highlights complaint/audit heterogeneity in cross-border settings.",
        "5. Within-authority contrasts expose private–public fining asymmetries for follow-up review.",
    ]

    common.write_summary(out_dir, summary_lines)
    common.write_memo(out_dir, memo_lines)
    common.write_session_info(out_dir)

    print("Task 3 completed. Key metrics:")
    for line in summary_lines[1:]:
        print(f"  - {line}")

    return out_dir


if __name__ == "__main__":
    run()
