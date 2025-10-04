"""Phase 5 – Cohort similarity and outcomes analysis.

This module operationalises the analysis specification provided for
GDPR enforcement decisions.  It assumes that Phase 4 enrichment has
already produced `1_enriched_master.csv` and builds new analytical
artefacts under `outputs/phase5_analysis/`.

The script is organised into the following sub-phases:

Subphase 0 – Article set parsing
    • Parse `a77_articles_breached` into numeric sets
    • Build deterministic keys for exact and family level cohorts

Subphase 1 – Baseline similarity within article cohorts
    • Exact article-set cohorts with summary outcomes
    • Relaxed cohorts using a Jaccard ≥ 0.8 threshold on article sets

Subphase 2 – Layered factor tests (contexts, legal bases, defendant type)
    • Pairwise contrasts inside frozen article cohorts while matching on
      all non-tested layers

Subphase 3 – Cross-country nearest-neighbour comparisons
    • Greedy matching of decisions across countries within article
      cohorts while holding role, sector, contexts, and Art. 83 profiles

Subphase 4 – Mixed-effects regression on log fines (2025 EUR)
    • Random intercept for article set and variance component for country

Subphase 5 – Robustness diagnostics
    • Leave-one-layer-out models, relaxed-cohort repeats, time controls

The resulting CSV and TXT artefacts are written to
`outputs/phase5_analysis/`.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


BASE_DATA_PATH = Path("outputs/phase4_enrichment/1_enriched_master.csv")
OUTPUT_DIR = Path("outputs/phase5_analysis")


# Mapping between context boolean features and their canonical string label
# inside `context_profile` / `a14_processing_contexts`.
CONTEXT_FLAG_MAP = {
    "has_employee_monitoring": "EMPLOYEE_MONITORING",
    "has_cctv": "CCTV",
    "has_marketing": "MARKETING",
    "has_problematic_third_party_sharing": "PROBLEMATIC_THIRD_PARTY_SHARING_STATED",
    "has_recruitment_hr": "RECRUITMENT_HR",
    "has_cookies": "COOKIES",
    "has_ai": "AI",
    "has_credit_scoring": "CREDIT_SCORING",
}


MEASURE_COL_MAP = {
    "a45_warning_issued_bool": "WARNING",
    "a46_reprimand_issued_bool": "REPRIMAND",
    "a47_comply_data_subject_order_bool": "DATA_SUBJECT_ORDER",
    "a48_compliance_order_bool": "COMPLIANCE_ORDER",
    "a49_breach_communication_order_bool": "BREACH_COMMUNICATION_ORDER",
    "a50_erasure_restriction_order_bool": "ERASURE_OR_RESTRICTION_ORDER",
    "a51_certification_withdrawal_bool": "CERTIFICATION_WITHDRAWAL",
    "a52_data_flow_suspension_bool": "DATA_FLOW_SUSPENSION",
}


ARTICLE_FAMILY_MAP: Dict[int, str] = {
    2: "2",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    12: "12-15",
    13: "12-15",
    14: "12-15",
    15: "12-15",
    16: "16-18",
    17: "16-18",
    18: "16-18",
    21: "21-22",
    22: "21-22",
    24: "24-26",
    25: "24-26",
    26: "24-26",
    27: "27",
    28: "28-29",
    29: "28-29",
    30: "30-31",
    31: "30-31",
    32: "32-34",
    33: "32-34",
    34: "32-34",
    35: "35",
    37: "37-44",
    38: "37-44",
    39: "37-44",
    40: "37-44",
    41: "37-44",
    42: "37-44",
    43: "37-44",
    44: "37-44",
    46: "46",
    58: "58",
    66: "66",
    82: "82",
    83: "83",
    89: "89-90",
    90: "89-90",
    104: "104",
    130: "130",
}


ROBUSTNESS_MODEL_VARIANTS = {
    "full": {
        "drop_contexts": False,
        "drop_legal_basis": False,
    },
    "no_contexts": {
        "drop_contexts": True,
        "drop_legal_basis": False,
    },
    "no_legal_basis": {
        "drop_contexts": False,
        "drop_legal_basis": True,
    },
}


ART83_SCORE_COLS = [
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
    "art83_balance_score",
]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_master() -> pd.DataFrame:
    if not BASE_DATA_PATH.exists():
        raise FileNotFoundError(
            "Phase 4 master dataset is missing – run enrichment before analysis."
        )
    df = pd.read_csv(BASE_DATA_PATH)
    return df


ARTICLE_REGEX = re.compile(r"\d+")


def parse_articles(value: str) -> Tuple[int, ...]:
    if pd.isna(value):
        return tuple()
    text = str(value).strip()
    if text in {"", "NOT_DISCUSSED", "NOT_APPLICABLE", "NONE_VIOLATED"}:
        return tuple()
    numbers = {int(match.group(0)) for match in ARTICLE_REGEX.finditer(text)}
    return tuple(sorted(numbers))


def build_family_key(articles: Iterable[int]) -> str:
    families = {ARTICLE_FAMILY_MAP.get(num, str(num)) for num in articles}
    if not families:
        return "NONE"
    return ";".join(sorted(families))


def measure_set(row: pd.Series) -> frozenset[str]:
    active = [name for col, name in MEASURE_COL_MAP.items() if bool(row.get(col, False))]
    return frozenset(active)


def average_jaccard(sets: Sequence[frozenset[str]]) -> float:
    if len(sets) < 2:
        return float("nan")
    totals: List[float] = []
    for left, right in itertools.combinations(sets, 2):
        union = left | right
        if not union:
            totals.append(1.0)
            continue
        totals.append(len(left & right) / len(union))
    return float(np.mean(totals)) if totals else float("nan")


def prepare_case_level(df: pd.DataFrame) -> pd.DataFrame:
    parsed_articles = df["a77_articles_breached"].apply(parse_articles)
    article_keys = [";".join(map(str, arts)) if arts else "NONE" for arts in parsed_articles]
    family_keys = [build_family_key(arts) for arts in parsed_articles]

    measures = df.apply(measure_set, axis=1)

    case_df = df.copy()
    case_df["article_set"] = parsed_articles
    case_df["article_set_key"] = article_keys
    case_df["article_family_key"] = family_keys
    case_df["measure_set"] = measures
    case_df["measure_set_size"] = measures.apply(len)
    case_df["log_fine_2025"] = np.log1p(
        case_df["fine_amount_eur_real_2025"].clip(lower=0).fillna(0.0)
    )
    case_df["a4_decision_year"] = pd.to_numeric(
        case_df["a4_decision_year"], errors="coerce"
    )

    def simplify_defendant(value: str) -> str:
        if value in {"PRIVATE", "PUBLIC"}:
            return value
        if value == "INDIVIDUAL":
            return "INDIVIDUAL"
        if value in {"NOT_DISCUSSSED", "NOT_DISCUSSED", "NOT_APPLICABLE", ""}:
            return "UNSPECIFIED"
        return "OTHER"

    case_df["defendant_type_slim"] = case_df["a8_defendant_class"].apply(simplify_defendant)

    for flag, label in CONTEXT_FLAG_MAP.items():
        if flag not in case_df.columns:
            continue
        profile_without = []
        for original, has_flag in zip(case_df["context_profile"], case_df[flag].fillna(False)):
            items: List[str]
            if isinstance(original, str) and original.strip():
                items = [part for part in original.split(";") if part]
            else:
                items = []
            filtered = [part for part in items if part != label]
            profile_without.append(";".join(sorted(filtered)) if filtered else "NONE")
        case_df[f"stratum_without_{flag}"] = profile_without

    return case_df


def summarise_baseline(case_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for key, group in case_df.groupby("article_set_key"):
        measures = group["measure_set"].tolist()
        record = {
            "article_set_key": key,
            "article_family_key": build_family_key(group.iloc[0]["article_set"]),
            "article_count": len(group.iloc[0]["article_set"]),
            "case_count": len(group),
            "country_count": group["a1_country_code"].nunique(dropna=True),
            "log_fine_mean": group["log_fine_2025"].mean(),
            "log_fine_std": group["log_fine_2025"].std(ddof=0),
            "log_fine_median": group["log_fine_2025"].median(),
            "measure_jaccard_mean": average_jaccard(measures),
            "measure_jaccard_median": np.nanmedian(
                [
                    len(left & right) / len(left | right)
                    if (left | right)
                    else 1.0
                    for left, right in itertools.combinations(measures, 2)
                ]
            )
            if len(measures) > 1
            else float("nan"),
            "sanction_profile_mode": group["sanction_profile"].mode(dropna=True).iat[0]
            if not group["sanction_profile"].mode(dropna=True).empty
            else "UNSPECIFIED",
        }
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values("case_count", ascending=False)


class UnionFind:
    def __init__(self, elements: Iterable[str]):
        self.parent = {element: element for element in elements}

    def find(self, item: str) -> str:
        parent = self.parent.setdefault(item, item)
        if parent != item:
            parent = self.find(parent)
        self.parent[item] = parent
        return parent

    def union(self, left: str, right: str) -> None:
        self.parent[self.find(left)] = self.find(right)


def relaxed_components(case_df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    unique_sets = (
        case_df[["article_set_key", "article_set"]]
        .drop_duplicates()
        .set_index("article_set_key")
    )
    union_find = UnionFind(unique_sets.index)

    for key_left, key_right in itertools.combinations(unique_sets.index, 2):
        set_left = set(unique_sets.loc[key_left, "article_set"])
        set_right = set(unique_sets.loc[key_right, "article_set"])
        if not set_left and not set_right:
            similarity = 1.0
        else:
            union = set_left | set_right
            if not union:
                similarity = 1.0
            else:
                similarity = len(set_left & set_right) / len(union)
        if similarity >= threshold:
            union_find.union(key_left, key_right)

    component_map = {key: union_find.find(key) for key in unique_sets.index}
    case_df = case_df.copy()
    case_df["article_component_key"] = case_df["article_set_key"].map(component_map)

    records: List[Dict[str, object]] = []
    for component, group in case_df.groupby("article_component_key"):
        measures = group["measure_set"].tolist()
        article_sets = sorted(group["article_set_key"].unique())
        record = {
            "article_component_key": component,
            "member_article_sets": ";".join(article_sets),
            "case_count": len(group),
            "article_count_median": group["article_set"].apply(len).median(),
            "article_family_keys": ";".join(sorted(group["article_family_key"].unique())),
            "log_fine_mean": group["log_fine_2025"].mean(),
            "measure_jaccard_mean": average_jaccard(measures),
            "sanction_profile_mode": group["sanction_profile"].mode(dropna=True).iat[0]
            if not group["sanction_profile"].mode(dropna=True).empty
            else "UNSPECIFIED",
        }
        records.append(record)

    summary = pd.DataFrame.from_records(records).sort_values("case_count", ascending=False)
    return case_df, summary


def _context_stratum_key(row: pd.Series, flag: str) -> Tuple[str, str, str, str]:
    return (
        row[f"stratum_without_{flag}"] if f"stratum_without_{flag}" in row else "NONE",
        row.get("a11_defendant_role", "UNKNOWN"),
        row.get("a12_sector", "UNKNOWN"),
        row.get("defendant_type_slim", "UNSPECIFIED"),
    )


def _median(values: Sequence[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.median(values))


def _jaccard_series(sets: Sequence[frozenset[str]]) -> float:
    if not sets:
        return float("nan")
    return average_jaccard(sets)


def analyse_context_effects(case_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for flag in CONTEXT_FLAG_MAP:
        if flag not in case_df.columns:
            continue
        for key, cohort in case_df.groupby("article_set_key"):
            if cohort[flag].nunique() < 2:
                continue
            cohort = cohort.copy()
            cohort["_stratum"] = cohort.apply(
                _context_stratum_key, axis=1, args=(flag,)
            )
            for stratum, group in cohort.groupby("_stratum"):
                exposed = group[group[flag]]
                control = group[~group[flag]]
                if len(exposed) < 2 or len(control) < 2:
                    continue
                log_fine_exposed = exposed["log_fine_2025"].to_numpy()
                log_fine_control = control["log_fine_2025"].to_numpy()
                mannwhitney = stats.mannwhitneyu(log_fine_exposed, log_fine_control, alternative="two-sided")
                record = {
                    "factor": flag,
                    "article_set_key": key,
                    "stratum_context": stratum[0],
                    "stratum_role": stratum[1],
                    "stratum_sector": stratum[2],
                    "stratum_defendant": stratum[3],
                    "exposed_count": len(exposed),
                    "control_count": len(control),
                    "log_fine_exposed_median": _median(log_fine_exposed),
                    "log_fine_control_median": _median(log_fine_control),
                    "log_fine_median_diff": _median(log_fine_exposed) - _median(log_fine_control),
                    "mannwhitney_stat": mannwhitney.statistic,
                    "mannwhitney_pvalue": mannwhitney.pvalue,
                    "measure_jaccard_exposed": _jaccard_series(exposed["measure_set"].tolist()),
                    "measure_jaccard_control": _jaccard_series(control["measure_set"].tolist()),
                    "sanction_profile_mode_exposed": exposed["sanction_profile"].mode(dropna=True).iat[0]
                    if not exposed["sanction_profile"].mode(dropna=True).empty
                    else "UNSPECIFIED",
                    "sanction_profile_mode_control": control["sanction_profile"].mode(dropna=True).iat[0]
                    if not control["sanction_profile"].mode(dropna=True).empty
                    else "UNSPECIFIED",
                }
                records.append(record)
    return pd.DataFrame.from_records(records)


def _legal_basis_stratum(row: pd.Series, column: str) -> Tuple[str, str, str, str, str]:
    context_key = row.get("context_profile", "NONE") or "NONE"
    return (
        row.get("article_set_key", "NONE"),
        context_key,
        row.get("a11_defendant_role", "UNKNOWN"),
        row.get("a12_sector", "UNKNOWN"),
        row.get("defendant_type_slim", "UNSPECIFIED"),
    )


def analyse_legal_basis(case_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["a30_art6_consent", "a31_art6_contract", "a35_art6_legitimate_interest"]
    records: List[Dict[str, object]] = []
    for column in columns:
        if column not in case_df.columns:
            continue
        for key, cohort in case_df.groupby("article_set_key"):
            categories = cohort[column].fillna("NOT_DISCUSSED").unique()
            if "INVALID" not in categories:
                continue
            cohort = cohort.copy()
            cohort["_stratum"] = cohort.apply(
                _legal_basis_stratum, axis=1, args=(column,)
            )
            for stratum, group in cohort.groupby("_stratum"):
                invalid = group[group[column] == "INVALID"]
                remaining = group[group[column] != "INVALID"]
                if len(invalid) < 2 or remaining.empty:
                    continue
                for comparator_value, subset in remaining.groupby(column):
                    if len(subset) < 2:
                        continue
                    stat = stats.mannwhitneyu(
                        invalid["log_fine_2025"], subset["log_fine_2025"], alternative="two-sided"
                    )
                    record = {
                        "factor": column,
                        "comparator": comparator_value,
                        "article_set_key": key,
                        "stratum_context": stratum[1],
                        "stratum_role": stratum[2],
                        "stratum_sector": stratum[3],
                        "stratum_defendant": stratum[4],
                        "invalid_count": len(invalid),
                        "comparator_count": len(subset),
                        "log_fine_invalid_median": _median(invalid["log_fine_2025"]),
                        "log_fine_comparator_median": _median(subset["log_fine_2025"]),
                        "log_fine_median_diff": _median(invalid["log_fine_2025"]) - _median(subset["log_fine_2025"]),
                        "mannwhitney_stat": stat.statistic,
                        "mannwhitney_pvalue": stat.pvalue,
                        "measure_jaccard_invalid": _jaccard_series(invalid["measure_set"].tolist()),
                        "measure_jaccard_comparator": _jaccard_series(subset["measure_set"].tolist()),
                        "sanction_profile_mode_invalid": invalid["sanction_profile"].mode(dropna=True).iat[0]
                        if not invalid["sanction_profile"].mode(dropna=True).empty
                        else "UNSPECIFIED",
                        "sanction_profile_mode_comparator": subset["sanction_profile"].mode(dropna=True).iat[0]
                        if not subset["sanction_profile"].mode(dropna=True).empty
                        else "UNSPECIFIED",
                    }
                    records.append(record)
    return pd.DataFrame.from_records(records)


def analyse_defendant_type(case_df: pd.DataFrame) -> pd.DataFrame:
    context_labels = list(CONTEXT_FLAG_MAP.values())
    records: List[Dict[str, object]] = []
    for label in context_labels:
        subset = case_df[case_df["context_profile"].fillna("").str.contains(label, na=False)]
        if subset.empty:
            continue
        for key, cohort in subset.groupby("article_set_key"):
            cohort = cohort.copy()
            cohort["_stratum"] = list(
                zip(
                    cohort.get("context_profile", "NONE").fillna("NONE"),
                    cohort.get("a11_defendant_role", "UNKNOWN"),
                    cohort.get("a12_sector", "UNKNOWN"),
                )
            )
            for stratum, group in cohort.groupby("_stratum"):
                private = group[group["defendant_type_slim"] == "PRIVATE"]
                public = group[group["defendant_type_slim"] == "PUBLIC"]
                if len(private) < 2 or len(public) < 2:
                    continue
                stat = stats.mannwhitneyu(private["log_fine_2025"], public["log_fine_2025"], alternative="two-sided")
                record = {
                    "context_label": label,
                    "article_set_key": key,
                    "stratum_context": stratum[0],
                    "stratum_role": stratum[1],
                    "stratum_sector": stratum[2],
                    "private_count": len(private),
                    "public_count": len(public),
                    "log_fine_private_median": _median(private["log_fine_2025"]),
                    "log_fine_public_median": _median(public["log_fine_2025"]),
                    "log_fine_median_diff": _median(private["log_fine_2025"]) - _median(public["log_fine_2025"]),
                    "mannwhitney_stat": stat.statistic,
                    "mannwhitney_pvalue": stat.pvalue,
                    "measure_jaccard_private": _jaccard_series(private["measure_set"].tolist()),
                    "measure_jaccard_public": _jaccard_series(public["measure_set"].tolist()),
                    "sanction_profile_mode_private": private["sanction_profile"].mode(dropna=True).iat[0]
                    if not private["sanction_profile"].mode(dropna=True).empty
                    else "UNSPECIFIED",
                    "sanction_profile_mode_public": public["sanction_profile"].mode(dropna=True).iat[0]
                    if not public["sanction_profile"].mode(dropna=True).empty
                    else "UNSPECIFIED",
                }
                records.append(record)
    return pd.DataFrame.from_records(records)


@dataclass
class CrossCountryPair:
    article_set_key: str
    stratum_context: str
    stratum_role: str
    stratum_sector: str
    case_id_left: str
    country_left: str
    case_id_right: str
    country_right: str
    log_fine_left: float
    log_fine_right: float
    fine_imposed_left: bool
    fine_imposed_right: bool
    measure_jaccard: float


def _art83_vector(row: pd.Series) -> np.ndarray:
    values = (
        pd.to_numeric(row[ART83_SCORE_COLS], errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )
    return values


def analyse_cross_country(case_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairs: List[CrossCountryPair] = []
    for (article_key, context, role, sector), group in case_df.groupby(
        ["article_set_key", "context_profile", "a11_defendant_role", "a12_sector"]
    ):
        if group["a1_country_code"].nunique() < 2:
            continue
        available = list(group.index)
        used = set()
        while available:
            idx = available.pop(0)
            if idx in used:
                continue
            row = group.loc[idx]
            candidates = [
                other_idx
                for other_idx in group.index
                if other_idx not in used
                and other_idx != idx
                and group.loc[other_idx, "a1_country_code"] != row["a1_country_code"]
            ]
            if not candidates:
                continue
            row_vec = _art83_vector(row)
            distances = []
            for cand_idx in candidates:
                cand_vec = _art83_vector(group.loc[cand_idx])
                distances.append((np.linalg.norm(row_vec - cand_vec), cand_idx))
            _, best_idx = min(distances, key=lambda item: item[0])
            used.add(idx)
            used.add(best_idx)
            available = [a for a in available if a not in used]
            left_id, right_id = row["id"], group.loc[best_idx, "id"]
            left_measure = row["measure_set"]
            right_measure = group.loc[best_idx, "measure_set"]
            union = left_measure | right_measure
            measure_jaccard = (
                len(left_measure & right_measure) / len(union) if union else 1.0
            )
            pairs.append(
                CrossCountryPair(
                    article_set_key=article_key,
                    stratum_context=context or "NONE",
                    stratum_role=role,
                    stratum_sector=sector,
                    case_id_left=left_id,
                    country_left=row["a1_country_code"],
                    case_id_right=right_id,
                    country_right=group.loc[best_idx, "a1_country_code"],
                    log_fine_left=row["log_fine_2025"],
                    log_fine_right=group.loc[best_idx, "log_fine_2025"],
                    fine_imposed_left=bool(row.get("fine_imposed_bool", False)),
                    fine_imposed_right=bool(group.loc[best_idx].get("fine_imposed_bool", False)),
                    measure_jaccard=measure_jaccard,
                )
            )

    if not pairs:
        return pd.DataFrame(), pd.DataFrame()

    pair_df = pd.DataFrame([pair.__dict__ for pair in pairs])
    summaries: List[Dict[str, object]] = []
    for key, group in pair_df.groupby("article_set_key"):
        if len(group) < 2:
            continue
        diff = group["log_fine_left"] - group["log_fine_right"]
        ttest = stats.ttest_rel(group["log_fine_left"], group["log_fine_right"])
        b = int(((group["fine_imposed_left"] == 1) & (group["fine_imposed_right"] == 0)).sum())
        c = int(((group["fine_imposed_left"] == 0) & (group["fine_imposed_right"] == 1)).sum())
        if b + c > 0:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            pvalue = 1 - stats.chi2.cdf(chi2, df=1)
        else:
            chi2 = float("nan")
            pvalue = float("nan")
        summaries.append(
            {
                "article_set_key": key,
                "pair_count": len(group),
                "mean_log_fine_diff": diff.mean(),
                "std_log_fine_diff": diff.std(ddof=0),
                "paired_t_stat": ttest.statistic,
                "paired_t_pvalue": ttest.pvalue,
                "mcnemar_b": b,
                "mcnemar_c": c,
                "mcnemar_chi2": chi2,
                "mcnemar_pvalue": pvalue,
                "mean_measure_jaccard": group["measure_jaccard"].mean(),
            }
        )

    return pair_df, pd.DataFrame(summaries)


def _build_model_formula(drop_contexts: bool, drop_legal_basis: bool) -> str:
    contexts = [
        "has_employee_monitoring",
        "has_cctv",
        "has_marketing",
        "has_problematic_third_party_sharing",
    ]
    legal_basis_terms = [
        "C(a30_art6_consent)",
        "C(a31_art6_contract)",
        "C(a35_art6_legitimate_interest)",
    ]
    defendant_terms = [
        "C(defendant_type_slim)",
        "C(a9_enterprise_size)",
        "C(a11_defendant_role)",
    ]

    formula_parts: List[str] = ["a4_decision_year"]
    if not drop_contexts:
        formula_parts.extend(contexts)
    if not drop_legal_basis:
        formula_parts.extend(legal_basis_terms)
    formula_parts.extend(defendant_terms)
    rhs = " + ".join(formula_parts)
    return f"log_fine_2025 ~ {rhs}"


def run_mixed_models(case_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    analysis_df = case_df.copy()
    analysis_df = analysis_df.replace({"NOT_APPLICABLE": "NOT_DISCUSSED"})
    categorical_cols = [
        "a30_art6_consent",
        "a31_art6_contract",
        "a35_art6_legitimate_interest",
        "defendant_type_slim",
        "a9_enterprise_size",
        "a11_defendant_role",
    ]
    for col in categorical_cols:
        if col in analysis_df.columns:
            analysis_df[col] = analysis_df[col].fillna("UNSPECIFIED")
    context_cols = [
        "has_employee_monitoring",
        "has_cctv",
        "has_marketing",
        "has_problematic_third_party_sharing",
    ]
    for col in context_cols:
        if col in analysis_df.columns:
            analysis_df[col] = analysis_df[col].fillna(False)
    analysis_df = analysis_df.dropna(
        subset=["log_fine_2025", "article_set_key", "a1_country_code", "a4_decision_year"]
    )

    results: List[Dict[str, object]] = []
    summaries = {}
    for variant, options in ROBUSTNESS_MODEL_VARIANTS.items():
        formula = _build_model_formula(
            drop_contexts=options["drop_contexts"],
            drop_legal_basis=options["drop_legal_basis"],
        )
        model = smf.mixedlm(
            formula,
            analysis_df,
            groups=analysis_df["article_set_key"],
        )
        try:
            result = model.fit(method="lbfgs", maxiter=200)
        except Exception as exc:  # pragma: no cover - defensive
            results.append(
                {
                    "variant": variant,
                    "converged": False,
                    "message": str(exc),
                }
            )
            continue
        summaries[variant] = result.summary().as_text()
        fit_history = getattr(result, "fit_history", {})
        for param, estimate in result.params.items():
            results.append(
                {
                    "variant": variant,
                    "parameter": param,
                    "estimate": estimate,
                    "std_err": result.bse.get(param, float("nan")),
                    "pvalue": result.pvalues.get(param, float("nan")),
                    "converged": getattr(result, "converged", False),
                    "iterations": fit_history.get("iterations", np.nan),
                }
            )

    summary_text = json.dumps(summaries, indent=2)
    return pd.DataFrame(results), summary_text


def compare_exact_relaxed(context_results: pd.DataFrame, case_df: pd.DataFrame) -> pd.DataFrame:
    if context_results.empty:
        return pd.DataFrame()
    case_with_components, _ = relaxed_components(case_df)
    component_map = (
        case_with_components[["article_set_key", "article_component_key"]]
        .drop_duplicates()
        .set_index("article_set_key")
    )

    records: List[Dict[str, object]] = []
    for factor, group in context_results.groupby("factor"):
        component_keys = group["article_set_key"].map(component_map["article_component_key"])
        for component, comp_group in group.groupby(component_keys):
            if comp_group.empty or pd.isna(component):
                continue
            weights = comp_group["exposed_count"] + comp_group["control_count"]
            if weights.sum() == 0:
                continue
            records.append(
                {
                    "factor": factor,
                    "article_component_key": component,
                    "stratum_count": comp_group["stratum_context"].nunique(),
                    "weighted_median_diff": float(
                        np.average(comp_group["log_fine_median_diff"], weights=weights)
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def build_time_controls(case_df: pd.DataFrame) -> pd.DataFrame:
    case_df = case_df.copy()
    case_df["period_bucket"] = np.where(
        case_df["a4_decision_year"] >= 2021,
        "2021_plus",
        "pre_2021",
    )
    records: List[Dict[str, object]] = []
    for key, group in case_df.groupby(["article_set_key", "period_bucket"]):
        records.append(
            {
                "article_set_key": key[0],
                "period_bucket": key[1],
                "case_count": len(group),
                "log_fine_mean": group["log_fine_2025"].mean(),
                "measure_count_mean": group["measure_set_size"].mean(),
                "fine_imposed_rate": group.get("fine_imposed_bool", pd.Series(dtype=float)).mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    ensure_output_dir()
    master_df = load_master()
    case_df = prepare_case_level(master_df)
    case_df.to_csv(OUTPUT_DIR / "0_case_level_features.csv", index=False)

    baseline = summarise_baseline(case_df)
    baseline.to_csv(OUTPUT_DIR / "1_baseline_article_cohorts.csv", index=False)

    case_with_components, relaxed_summary = relaxed_components(case_df)
    case_with_components.to_csv(OUTPUT_DIR / "2_case_level_with_components.csv", index=False)
    relaxed_summary.to_csv(OUTPUT_DIR / "2_relaxed_article_components.csv", index=False)

    context_results = analyse_context_effects(case_df)
    if not context_results.empty:
        context_results.to_csv(OUTPUT_DIR / "3_context_effects.csv", index=False)

    legal_results = analyse_legal_basis(case_df)
    if not legal_results.empty:
        legal_results.to_csv(OUTPUT_DIR / "3_legal_basis_effects.csv", index=False)

    defendant_results = analyse_defendant_type(case_df)
    if not defendant_results.empty:
        defendant_results.to_csv(OUTPUT_DIR / "3_defendant_type_effects.csv", index=False)

    pair_df, cross_summary = analyse_cross_country(case_df)
    if not pair_df.empty:
        pair_df.to_csv(OUTPUT_DIR / "4_cross_country_pairs.csv", index=False)
    if not cross_summary.empty:
        cross_summary.to_csv(OUTPUT_DIR / "4_cross_country_summary.csv", index=False)

    model_results, model_summaries = run_mixed_models(case_df)
    model_results.to_csv(OUTPUT_DIR / "5_mixed_effects_results.csv", index=False)
    (OUTPUT_DIR / "5_mixed_effects_summary.txt").write_text(model_summaries)

    if not context_results.empty:
        relaxed_contrast = compare_exact_relaxed(context_results, case_df)
        if not relaxed_contrast.empty:
            relaxed_contrast.to_csv(OUTPUT_DIR / "6_relaxed_cohort_contrasts.csv", index=False)

    time_controls = build_time_controls(case_df)
    time_controls.to_csv(OUTPUT_DIR / "6_time_controls_summary.csv", index=False)


if __name__ == "__main__":
    main()

