#!/usr/bin/env python3
"""Phase 4: Enrich Phase 3 output and emit analyst-ready datasets."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SENTINEL_VALUES = {
    "", "NOT_APPLICABLE", "NOT_DISCUSSED", "NONE", "UNKNOWN", "NOT_AVAILABLE",
}

ART5_FIELDS = {
    "a21_art5_lawfulness_fairness": "art5_1_a",
    "a22_art5_purpose_limitation": "art5_1_b",
    "a23_art5_data_minimization": "art5_1_c",
    "a24_art5_accuracy": "art5_1_d",
    "a25_art5_storage_limitation": "art5_1_e",
    "a26_art5_integrity_confidentiality": "art5_1_f",
    "a27_art5_accountability": "art5_2",
}

RIGHT_FIELDS = {
    "a37_right_access_violated": "A15",
    "a38_right_rectification_violated": "A16",
    "a39_right_erasure_violated": "A17",
    "a40_right_restriction_violated": "A18",
    "a41_right_portability_violated": "A20",
    "a42_right_object_violated": "A21",
    "a43_transparency_violated": "A12",
    "a44_automated_decisions_violated": "A22",
}

RIGHT_TO_ARTICLE = {
    "a37_right_access_violated": 15,
    "a38_right_rectification_violated": 16,
    "a39_right_erasure_violated": 17,
    "a40_right_restriction_violated": 18,
    "a41_right_portability_violated": 20,
    "a42_right_object_violated": 21,
    "a43_transparency_violated": 12,
    "a44_automated_decisions_violated": 22,
}

ART83_FIELDS = [
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
]

MEASURE_FIELDS = [
    "a45_warning_issued",
    "a46_reprimand_issued",
    "a47_comply_data_subject_order",
    "a48_compliance_order",
    "a49_breach_communication_order",
    "a50_erasure_restriction_order",
    "a51_certification_withdrawal",
    "a52_data_flow_suspension",
]

LEGAL_KEYWORDS: Dict[str, Sequence[str]] = {
    "invalid consent": ("invalid consent", "consent not valid", "lack of consent"),
    "contract": ("contractual necessity", "contract basis", "performance of a contract"),
    "public task": ("public task", "public interest task"),
    "legitimate interest invalid": ("illegitimate interest", "invalid legitimate interest", "unbalanced legitimate interest"),
    "cookies": ("cookie", "cookies"),
}


def load_context_taxonomy(path: Path) -> pd.DataFrame:
    """Load processing context taxonomy with priority and flag column metadata."""

    taxonomy = pd.read_csv(path)
    expected_columns = {"processing_context", "priority_rank", "flag_column"}
    missing = expected_columns.difference(taxonomy.columns)
    if missing:
        raise ValueError(
            f"Context taxonomy at {path} missing required columns: {', '.join(sorted(missing))}"
        )
    if taxonomy.empty:
        raise ValueError(f"Context taxonomy at {path} is empty; at least one entry is required")

    taxonomy["processing_context"] = taxonomy["processing_context"].astype(str).str.strip()
    taxonomy["flag_column"] = taxonomy["flag_column"].astype(str).str.strip()
    taxonomy["priority_rank"] = taxonomy["priority_rank"].astype(int)
    taxonomy = taxonomy.sort_values("priority_rank").reset_index(drop=True)
    return taxonomy


def load_region_map(path: Path) -> Dict[str, str]:
    """Load country-to-region mappings from a CSV reference."""

    mapping_df = pd.read_csv(path)
    expected_columns = {"country_code", "region"}
    missing = expected_columns.difference(mapping_df.columns)
    if missing:
        raise ValueError(
            f"Region map at {path} missing required columns: {', '.join(sorted(missing))}"
        )
    mapping: Dict[str, str] = {}
    for _, row in mapping_df.iterrows():
        code = str(row["country_code"]).strip().upper()
        if not code:
            continue
        mapping[code] = str(row["region"]).strip()
    if not mapping:
        raise ValueError(f"Region map at {path} produced no usable rows")
    return mapping

ARTICLE_PRIORITY = [5, 6, 32, 13, 15, 21, 33, 34]
TARGET_ARTICLE_FLAGS = [5, 6, 13, 15, 21, 32, 33, 34]
DEFENDANT_CLASSES_NO_TURNOVER = {"PUBLIC", "INDIVIDUAL"}


def clean_sentinel(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text.upper() in SENTINEL_VALUES:
        return None
    return text


def parse_numeric(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = clean_sentinel(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return None


def parse_float(value: Optional[str]) -> Optional[float]:
    text = clean_sentinel(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def yes_no_to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    text = value.strip().upper()
    if text in {"YES", "BREACHED", "PRESENT"}:
        return True
    if text in {"NO", "NOT_BREACHED", "ABSENT"}:
        return False
    return None


def map_art83_value(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = value.strip().upper()
    if text == "AGGRAVATING":
        return 1
    if text == "MITIGATING":
        return -1
    if text == "NEUTRAL":
        return 0
    return None


def explode_semicolon_list(df: pd.DataFrame, column: str, value_name: str) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for _, row in df[["id", column]].iterrows():
        raw_value = row[column]
        if pd.isna(raw_value):
            continue
        tokens = [tok.strip() for tok in str(raw_value).split(";")]
        position = 1
        for token in tokens:
            if not token or token.upper() in SENTINEL_VALUES:
                continue
            records.append({"id": row["id"], value_name: token, "position": position})
            position += 1
    return pd.DataFrame.from_records(records)


def parse_articles(df: pd.DataFrame) -> pd.DataFrame:
    article_records: List[Dict[str, object]] = []
    pattern = re.compile(r"\d+")
    for _, row in df[["id", "a77_articles_breached"]].iterrows():
        raw_value = clean_sentinel(row["a77_articles_breached"])
        if not raw_value:
            continue
        for position, token in enumerate([t.strip() for t in raw_value.split(";") if t.strip()], start=1):
            matches = list(pattern.finditer(token))
            if not matches:
                continue
            first_match = matches[0]
            article_number = int(first_match.group())
            trailing_text = token[first_match.end():].strip()
            extra_numeric_tokens = [m.group() for m in matches[1:]]
            detail_truncated = bool(extra_numeric_tokens) or bool(re.search(r"[A-Za-z]", trailing_text))
            article_records.append(
                {
                    "id": row["id"],
                    "article_reference": token,
                    "article_number": article_number,
                    "article_label": f"GDPR_{article_number}",
                    "position": position,
                    "article_reference_detail": trailing_text or None,
                    "article_detail_tokens": ";".join(extra_numeric_tokens) if extra_numeric_tokens else None,
                    "article_detail_truncated": detail_truncated,
                }
            )
    return pd.DataFrame.from_records(article_records)


def load_reference_tables(fx_path: Path, hicp_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    fx_rates = pd.read_csv(fx_path)
    fx_rates["currency"] = fx_rates["currency"].str.upper()
    fx_rates["year"] = fx_rates["year"].astype(int)
    fx_rates["month"] = fx_rates["month"].astype(int)
    fx_rates["fx_to_eur"] = fx_rates["fx_to_eur"].astype(float)

    hicp = pd.read_csv(hicp_path).set_index("year")["hicp_index"].astype(float)
    return fx_rates, hicp


def lookup_fx_rate(
    fx_rates: pd.DataFrame, currency: Optional[str], year: Optional[int], month: Optional[int]
) -> Tuple[Optional[float], Optional[str], Optional[int], Optional[int]]:
    if currency is None or year is None:
        return None, None, None, None

    currency = currency.upper()
    if currency == "EUR":
        return 1.0, "ANNUAL_AVG", year, 0

    month_val = month if month and 1 <= month <= 12 else None

    query = fx_rates[(fx_rates["currency"] == currency) & (fx_rates["year"] == year)]
    if month_val is not None:
        monthly = query[query["month"] == month_val]
        if not monthly.empty:
            rate = float(monthly.iloc[0]["fx_to_eur"])
            return rate, "MONTHLY_AVG", year, month_val

    annual = query[query["month"] == 0]
    if not annual.empty:
        rate = float(annual.iloc[0]["fx_to_eur"])
        return rate, "ANNUAL_AVG", year, 0

    fallback = fx_rates[(fx_rates["currency"] == currency)]
    if not fallback.empty:
        rate = float(fallback.iloc[-1]["fx_to_eur"])
        source_year = int(fallback.iloc[-1]["year"])
        source_month = int(fallback.iloc[-1]["month"])
        return rate, "FALLBACK", source_year, source_month

    return None, None, None, None


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    decision_year = df["a4_decision_year"].apply(parse_numeric).astype("Int64")
    decision_month = df["a5_decision_month"].apply(parse_numeric).fillna(0)
    decision_month = decision_month.where(decision_month.between(1, 12), 0).astype("Int64")

    df["decision_year"] = decision_year
    df["decision_month"] = decision_month
    df["temporal_granularity"] = np.where(decision_month > 0, "YEAR_MONTH", "YEAR_ONLY")
    df["year_mon_known"] = decision_month > 0

    inferred_dates = []
    for year, month in zip(decision_year.tolist(), decision_month.tolist()):
        if pd.isna(year) or month == 0 or pd.isna(month):
            inferred_dates.append(pd.NaT)
        else:
            inferred_dates.append(pd.Timestamp(year=int(year), month=int(month), day=1))
    df["decision_date_inferred"] = inferred_dates

    df["decision_qtr"] = (
        df["decision_month"].where(df["decision_month"] > 0)
        .apply(lambda m: int(math.ceil(m / 3)) if pd.notna(m) else pd.NA)
        .astype("Int64")
    )
    return df


def convert_monetary_columns(df: pd.DataFrame, fx_rates: pd.DataFrame, hicp_index: pd.Series) -> pd.DataFrame:
    df["fine_imposed_bool"] = df["a53_fine_imposed"].apply(yes_no_to_bool)
    df["fine_amount_orig"] = df["a54_fine_amount"].apply(parse_float)
    df["fine_currency"] = df["a55_fine_currency"].apply(clean_sentinel)
    df["turnover_discussed_bool"] = df["a56_turnover_discussed"].apply(yes_no_to_bool)
    df["turnover_amount_orig"] = df["a57_turnover_amount"].apply(parse_float)
    df["turnover_currency"] = df["a58_turnover_currency"].apply(clean_sentinel)

    fine_amount_eur: List[Optional[float]] = []
    turnover_amount_eur: List[Optional[float]] = []
    fx_methods: List[Optional[str]] = []
    fx_years: List[Optional[int]] = []
    fx_months: List[Optional[int]] = []
    fine_fx_fallback_flags: List[bool] = []
    turnover_fx_methods: List[Optional[str]] = []
    turnover_fx_years: List[Optional[int]] = []
    turnover_fx_months: List[Optional[int]] = []
    turnover_fx_fallback_flags: List[bool] = []

    for _, row in df.iterrows():
        year = int(row["decision_year"]) if pd.notna(row["decision_year"]) else None
        month = int(row["decision_month"]) if pd.notna(row["decision_month"]) else None

        amount = row["fine_amount_orig"]
        currency = row["fine_currency"]
        if amount is not None and currency is not None:
            rate, method, rate_year, rate_month = lookup_fx_rate(fx_rates, currency, year, month)
            if rate is not None:
                fine_amount_eur.append(amount * rate)
                fx_methods.append(method)
                fx_years.append(rate_year)
                fx_months.append(rate_month)
                fine_fx_fallback_flags.append(method == "FALLBACK")
            else:
                fine_amount_eur.append(None)
                fx_methods.append(None)
                fx_years.append(None)
                fx_months.append(None)
                fine_fx_fallback_flags.append(False)
        else:
            fine_amount_eur.append(None)
            fx_methods.append(None)
            fx_years.append(None)
            fx_months.append(None)
            fine_fx_fallback_flags.append(False)

        turnover_amount = row["turnover_amount_orig"]
        turnover_currency = row["turnover_currency"]
        if turnover_amount is not None and turnover_currency is not None:
            rate, method, rate_year, rate_month = lookup_fx_rate(fx_rates, turnover_currency, year, month)
            if rate is not None:
                turnover_amount_eur.append(turnover_amount * rate)
                turnover_fx_methods.append(method)
                turnover_fx_years.append(rate_year)
                turnover_fx_months.append(rate_month)
                turnover_fx_fallback_flags.append(method == "FALLBACK")
            else:
                turnover_amount_eur.append(None)
                turnover_fx_methods.append(None)
                turnover_fx_years.append(None)
                turnover_fx_months.append(None)
                turnover_fx_fallback_flags.append(False)
        else:
            turnover_amount_eur.append(None)
            turnover_fx_methods.append(None)
            turnover_fx_years.append(None)
            turnover_fx_months.append(None)
            turnover_fx_fallback_flags.append(False)

    df["fine_amount_eur"] = fine_amount_eur
    df["turnover_amount_eur"] = turnover_amount_eur
    df["fine_fx_method"] = fx_methods
    df["fine_fx_year"] = pd.Series(fx_years, dtype="Int64")
    df["fine_fx_month"] = pd.Series(fx_months, dtype="Int64")
    df["turnover_fx_method"] = turnover_fx_methods
    df["turnover_fx_year"] = pd.Series(turnover_fx_years, dtype="Int64")
    df["turnover_fx_month"] = pd.Series(turnover_fx_months, dtype="Int64")
    df["flag_fine_fx_fallback"] = fine_fx_fallback_flags
    df["flag_turnover_fx_fallback"] = turnover_fx_fallback_flags

    df["fine_present"] = df["fine_imposed_bool"]
    df["turnover_present"] = df["turnover_discussed_bool"] & df["turnover_amount_eur"].notna()

    df["fine_amount_log"] = df["fine_amount_eur"].apply(lambda v: math.log1p(v) if v is not None and v > 0 else None)

    def compute_ratio(row: pd.Series) -> Optional[float]:
        if (
            row["fine_amount_eur"] is None
            or row["turnover_amount_eur"] in (None, 0)
            or row["a8_defendant_class"] in DEFENDANT_CLASSES_NO_TURNOVER
        ):
            return None
        return row["fine_amount_eur"] / row["turnover_amount_eur"]

    df["fine_pct_turnover"] = df.apply(compute_ratio, axis=1)

    def assign_bucket(value: Optional[float]) -> Optional[str]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if value == 0:
            return "0"
        if value < 1_000:
            return "1-999"
        if value < 10_000:
            return "1k-9.9k"
        if value < 100_000:
            return "10k-99k"
        if value < 1_000_000:
            return "100k-999k"
        return "≥1m"

    df["fine_bucket_eur"] = df["fine_amount_eur"].apply(assign_bucket)
    df["turnover_bucket_eur"] = df["turnover_amount_eur"].apply(assign_bucket)

    def deflate(value: Optional[float], year: Optional[int]) -> Optional[float]:
        if value is None or year is None or year not in hicp_index.index:
            return None
        base = hicp_index.get(year)
        target = hicp_index.get(2025)
        if base in (None, 0) or target is None:
            return None
        return value * (target / base)

    df["fine_amount_eur_real_2025"] = [
        deflate(val, year if not pd.isna(year) else None)
        for val, year in zip(df["fine_amount_eur"], df["decision_year"])
    ]
    df["turnover_amount_eur_real_2025"] = [
        deflate(val, year if not pd.isna(year) else None)
        for val, year in zip(df["turnover_amount_eur"], df["decision_year"])
    ]

    return df


def compute_art5_and_rights(df: pd.DataFrame, articles_df: pd.DataFrame) -> pd.DataFrame:
    for column, suffix in ART5_FIELDS.items():
        df[f"{suffix}_breached_bool"] = df[column].apply(yes_no_to_bool)

    rights_bool_map = {
        column: df[column].apply(yes_no_to_bool) for column in RIGHT_FIELDS
    }
    for column, series in rights_bool_map.items():
        df[f"{column}_bool"] = series

    rights_bool_df = pd.DataFrame(rights_bool_map)
    df["rights_violated_count"] = rights_bool_df.apply(
        lambda row: sum(val is True for val in row), axis=1
    ).astype("Int64")

    def build_profile(row: pd.Series) -> Optional[str]:
        selected = [
            code
            for column, code in RIGHT_FIELDS.items()
            if row.get(column) is True
        ]
        if not selected:
            return None
        return ";".join(selected)

    df["rights_profile"] = rights_bool_df.apply(build_profile, axis=1)

    article_groups = (
        articles_df.groupby("id")["article_number"].apply(lambda series: sorted(set(series)))
        if not articles_df.empty
        else pd.Series(dtype=object)
    )

    df["breach_count_total"] = df["id"].map(lambda _id: len(article_groups.get(_id, []))).astype("Int64")

    def has_article(article_list: List[int], target: int) -> bool:
        return target in article_list if article_list else False

    for target in TARGET_ARTICLE_FLAGS:
        df[f"breach_has_art{target}"] = df["id"].map(lambda _id: has_article(article_groups.get(_id, []), target))

    def top_family(article_list: List[int]) -> Optional[str]:
        if not article_list:
            return None
        for target in ARTICLE_PRIORITY:
            if target in article_list:
                return f"GDPR_{target}"
        return "OTHER"

    df["breach_family_top"] = df["id"].map(lambda _id: top_family(article_groups.get(_id, [])))

    article_lookup = {row["id"]: article_groups.get(row["id"], []) for _, row in df.iterrows()}

    def art5_flag(row: pd.Series) -> bool:
        return any(row[f"{suffix}_breached_bool"] for suffix in ART5_FIELDS.values())

    df["breach_has_art5"] = df.apply(art5_flag, axis=1)

    def flag_rights_gap(row: pd.Series) -> bool:
        rights_triggered = [
            RIGHT_TO_ARTICLE[col]
            for col in RIGHT_FIELDS
            if row.get(f"{col}_bool") is True
        ]
        if not rights_triggered:
            return False
        article_numbers = article_lookup.get(row["id"], [])
        return not set(rights_triggered).issubset(article_numbers)

    df["flag_articles_vs_rights_gap"] = df.apply(flag_rights_gap, axis=1)

    def flag_art5_gap(row: pd.Series) -> bool:
        if not row.get("breach_has_art5"):
            return False
        articles = article_lookup.get(row["id"], [])
        return 5 not in articles

    df["flag_art5_breached_but_5_not_in_77"] = df.apply(flag_art5_gap, axis=1)

    return df


def compute_measures(df: pd.DataFrame) -> pd.DataFrame:
    measure_bools = {field: df[field].apply(yes_no_to_bool) for field in MEASURE_FIELDS}

    def as_bool(series: pd.Series) -> pd.Series:
        return series.apply(lambda value: bool(value) if value is True else False)

    for field, series in measure_bools.items():
        df[f"{field}_bool"] = series

    measure_bool_df = pd.DataFrame({field: as_bool(series) for field, series in measure_bools.items()})
    fine_bool = as_bool(df["fine_imposed_bool"])

    df["measure_any_bool"] = fine_bool | measure_bool_df.any(axis=1)
    df["measure_count"] = measure_bool_df.sum(axis=1).astype("Int64")

    def sanction_profile(row: pd.Series) -> str:
        if yes_no_to_bool(row.get("a52_data_flow_suspension")):
            return "Suspension"
        if yes_no_to_bool(row.get("a51_certification_withdrawal")):
            return "CertificationWithdrawal"
        if yes_no_to_bool(row.get("a50_erasure_restriction_order")):
            return "ErasureOrRestrictionOrder"
        if yes_no_to_bool(row.get("a48_compliance_order")):
            return "ComplianceOrder"
        if yes_no_to_bool(row.get("a46_reprimand_issued")):
            return "Reprimand"
        if yes_no_to_bool(row.get("a45_warning_issued")):
            return "Warning"
        return "NoFormalMeasure"

    df["sanction_profile"] = df.apply(sanction_profile, axis=1)

    def is_warning_only(row: pd.Series) -> bool:
        warning = yes_no_to_bool(row.get("a45_warning_issued"))
        other_measures = any(
            yes_no_to_bool(row.get(field)) for field in MEASURE_FIELDS if field != "a45_warning_issued"
        )
        fine = yes_no_to_bool(row.get("a53_fine_imposed"))
        return bool(warning and not other_measures and not fine)

    def is_fine_only(row: pd.Series) -> bool:
        fine = yes_no_to_bool(row.get("a53_fine_imposed"))
        other_measures = any(yes_no_to_bool(row.get(field)) for field in MEASURE_FIELDS)
        return bool(fine and not other_measures)

    df["is_warning_only"] = df.apply(is_warning_only, axis=1)
    df["is_fine_only"] = df.apply(is_fine_only, axis=1)
    return df


def compute_art83_scores(df: pd.DataFrame) -> pd.DataFrame:
    score_columns = {}
    for field in ART83_FIELDS:
        score_columns[field] = df[field].apply(map_art83_value)
        df[f"{field}_score"] = score_columns[field]

    scores_df = pd.DataFrame(score_columns)
    df["art83_balance_score"] = scores_df.sum(axis=1, skipna=True)
    df["art83_aggravating_count"] = scores_df.apply(lambda row: (row == 1).sum(), axis=1)
    df["art83_mitigating_count"] = scores_df.apply(lambda row: (row == -1).sum(), axis=1)
    df["art83_neutral_count"] = scores_df.apply(lambda row: (row == 0).sum(), axis=1)
    df["art83_discussed_count"] = scores_df.notna().sum(axis=1)
    df["art83_systematic_bool"] = df["a70_systematic_art83_discussion"].apply(yes_no_to_bool)

    def map_first_violation(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip().upper()
        if text == "YES":
            return "FIRST_TIME"
        if text == "NO":
            return "REPEAT"
        if text in SENTINEL_VALUES:
            return "NOT_DISCUSSED"
        return text

    df["first_violation_status"] = df["a71_first_violation"].apply(map_first_violation)
    return df


def compute_context_features(
    df: pd.DataFrame,
    contexts_df: pd.DataFrame,
    context_taxonomy: pd.DataFrame,
) -> pd.DataFrame:
    flag_columns: Dict[str, str] = {
        row["processing_context"]: row["flag_column"]
        for _, row in context_taxonomy.iterrows()
    }
    priority_contexts: List[str] = context_taxonomy["processing_context"].tolist()

    configured_contexts = set(flag_columns.keys())

    if contexts_df.empty:
        for column in flag_columns.values():
            df[column] = False
        df["context_count"] = 0
        df["context_profile"] = None
        df["flag_context_token_unmapped"] = False
        df["context_unknown_tokens"] = None
        df["sector_x_context_key"] = df["a12_sector"].fillna("UNKNOWN") + "::NONE"
        df["role_sector_key"] = df["a11_defendant_role"].fillna("UNKNOWN") + "::" + df["a12_sector"].fillna("UNKNOWN")
        return df

    context_map = contexts_df.groupby("id")["processing_context"].apply(list)

    def has_context(id_value: str, target: str) -> bool:
        contexts = context_map.get(id_value, [])
        return target in contexts

    for context_name, column in flag_columns.items():
        df[column] = df["id"].map(lambda _id: has_context(_id, context_name))

    df["context_count"] = df["id"].map(lambda _id: len(context_map.get(_id, [])))

    def context_profile(_id: str) -> Optional[str]:
        contexts = sorted(set(context_map.get(_id, [])))
        if not contexts:
            return None
        return ";".join(contexts)

    df["context_profile"] = df["id"].map(context_profile)

    def unknown_tokens(_id: str) -> Optional[str]:
        tokens = sorted({ctx for ctx in context_map.get(_id, []) if ctx not in configured_contexts})
        if not tokens:
            return None
        return ";".join(tokens)

    df["context_unknown_tokens"] = df["id"].map(unknown_tokens)
    df["flag_context_token_unmapped"] = df["context_unknown_tokens"].notna()

    def top_context(_id: str) -> str:
        contexts = context_map.get(_id, [])
        if not contexts:
            return "NONE"
        for candidate in priority_contexts:
            if candidate in contexts:
                return candidate
        return contexts[0]

    df["sector_x_context_key"] = df["a12_sector"].fillna("UNKNOWN") + "::" + df["id"].map(top_context)
    df["role_sector_key"] = df["a11_defendant_role"].fillna("UNKNOWN") + "::" + df["a12_sector"].fillna("UNKNOWN")
    return df


def compute_complaint_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["has_complaint_bool"] = df["a15_data_subject_complaint"].apply(yes_no_to_bool)
    df["official_audit_bool"] = df["a17_official_audit"].apply(yes_no_to_bool)
    df["art33_discussed_bool"] = df["a18_art33_discussed"].apply(yes_no_to_bool)
    df["art33_breached_bool"] = df["a19_art33_breached"].apply(yes_no_to_bool)

    df["breach_notification_effect_num"] = df["a20_breach_notification_effect"].apply(map_art83_value)

    def art33_inconsistency(row: pd.Series) -> bool:
        discussed = row.get("a18_art33_discussed", "").strip().upper()
        breach = row.get("a19_art33_breached", "").strip().upper()
        return discussed == "NO" and breach in {"YES", "NO"}

    df["flag_art33_inconsistency"] = df.apply(art33_inconsistency, axis=1)
    return df


def compute_oss_and_geography(df: pd.DataFrame, region_map: Dict[str, str]) -> pd.DataFrame:
    df["oss_case_bool"] = df["a72_cross_border_oss"].apply(yes_no_to_bool)

    def oss_role_lead(row: pd.Series) -> bool:
        return bool(row["oss_case_bool"] and isinstance(row["a73_oss_role"], str) and row["a73_oss_role"].strip().upper() == "LEAD")

    def oss_role_concerned(row: pd.Series) -> bool:
        return bool(
            row["oss_case_bool"] and isinstance(row["a73_oss_role"], str) and row["a73_oss_role"].strip().upper() == "CONCERNED"
        )

    df["oss_role_lead_bool"] = df.apply(oss_role_lead, axis=1)
    df["oss_role_concerned_bool"] = df.apply(oss_role_concerned, axis=1)

    def oss_category(row: pd.Series) -> str:
        if row["oss_role_lead_bool"]:
            return "Lead"
        if row["oss_role_concerned_bool"]:
            return "Concerned"
        if row["oss_case_bool"] is False:
            return "NoOSS"
        return "Unknown"

    df["oss_case_category"] = df.apply(oss_category, axis=1)

    df["country_code"] = df["a1_country_code"].str.upper()

    def normalize_authority(value: Optional[str]) -> Optional[str]:
        text = clean_sentinel(value)
        if text is None:
            return None
        normalized = re.sub(r"\s+", " ", text).strip()
        replacements = {
            "AEPD": "Agencia Española de Protección de Datos",
            "CNIL": "Commission Nationale de l'Informatique et des Libertés",
        }
        lower = normalized.lower()
        for key, val in replacements.items():
            if lower == key.lower():
                return val
        return normalized

    df["authority_name_raw"] = df["a2_authority_name"]
    df["authority_name_norm"] = df["a2_authority_name"].apply(normalize_authority)

    df["region"] = df["country_code"].map(region_map)
    return df


def compute_text_features(df: pd.DataFrame, guidelines_df: pd.DataFrame) -> pd.DataFrame:
    for column in ["a36_legal_basis_summary", "a75_case_summary", "a76_art83_weighing_summary"]:
        df[f"{column}_len"] = df[column].fillna("").apply(len)

    keyword_results: List[List[str]] = []
    for _, row in df.iterrows():
        text_blobs = " ".join(
            str(row.get(col, "")) for col in ["a36_legal_basis_summary", "a75_case_summary", "a76_art83_weighing_summary"]
        ).lower()
        tags: List[str] = []
        for label, patterns in LEGAL_KEYWORDS.items():
            if any(pattern in text_blobs for pattern in patterns):
                tags.append(label)
        keyword_results.append(";".join(sorted(set(tags))) if tags else None)

    df["keywords_legal_basis"] = keyword_results

    guideline_counts = guidelines_df.groupby("id").size() if not guidelines_df.empty else pd.Series(dtype=int)
    df["mentions_guidelines_count"] = df["id"].map(lambda _id: int(guideline_counts.get(_id, 0)))
    return df


def compute_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["flag_sector_other_missing_detail"] = df.apply(
        lambda row: (
            isinstance(row.get("a12_sector"), str)
            and row["a12_sector"].strip().upper() == "OTHER"
            and not clean_sentinel(row.get("a13_sector_other"))
        ),
        axis=1,
    )

    df["flag_fine_currency_missing"] = df.apply(
        lambda row: (
            yes_no_to_bool(row.get("a53_fine_imposed"))
            and not clean_sentinel(row.get("a55_fine_currency"))
        ),
        axis=1,
    )

    def systematic_mismatch(row: pd.Series) -> bool:
        if yes_no_to_bool(row.get("a70_systematic_art83_discussion")):
            discussed = sum(1 for field in ART83_FIELDS if row.get(field) and row.get(field).upper() != "NOT_DISCUSSED")
            return discussed < 7
        return False

    df["flag_systematic83_mismatch"] = df.apply(systematic_mismatch, axis=1)
    return df


def build_graph_exports(
    enriched_df: pd.DataFrame,
    contexts_df: pd.DataFrame,
    guidelines_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    graph_dir = output_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    def decision_node_id(decision_id: str) -> str:
        return f"Decision|{decision_id}"

    def slugify(label: Optional[str], prefix: str, fallback: str) -> str:
        if not label:
            return f"{prefix}|{fallback}"
        slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
        slug = slug.upper() or fallback
        return f"{prefix}|{slug}"

    decision_nodes = enriched_df[["id", "decision_year", "country_code", "fine_amount_eur", "sanction_profile"]].copy()
    decision_nodes.insert(0, "id:ID", decision_nodes["id"].apply(decision_node_id))
    decision_nodes.insert(1, ":LABEL", "Decision")
    decision_nodes.to_csv(graph_dir / "nodes_decision.csv", index=False)

    authority_nodes = (
        enriched_df[["authority_name_norm", "country_code"]]
        .dropna(subset=["authority_name_norm"])
        .drop_duplicates()
        .copy()
    )
    authority_nodes.insert(
        0,
        "id:ID",
        authority_nodes["authority_name_norm"].apply(lambda name: slugify(name, "Authority", "UNKNOWN")),
    )
    authority_nodes.insert(1, ":LABEL", "Authority")
    authority_nodes.rename(columns={"authority_name_norm": "name"}, inplace=True)
    authority_nodes.to_csv(graph_dir / "nodes_authority.csv", index=False)

    defendant_nodes = (
        enriched_df[["a7_defendant_name", "a8_defendant_class", "a9_enterprise_size", "a12_sector"]]
        .dropna(subset=["a7_defendant_name"])
        .drop_duplicates()
        .copy()
    )
    defendant_nodes.insert(
        0,
        "id:ID",
        defendant_nodes["a7_defendant_name"].apply(lambda name: slugify(name, "Defendant", "UNKNOWN_DEFENDANT")),
    )
    defendant_nodes.insert(1, ":LABEL", "Defendant")
    defendant_nodes.rename(
        columns={
            "a7_defendant_name": "name",
            "a8_defendant_class": "defendant_class",
            "a9_enterprise_size": "enterprise_size",
            "a12_sector": "sector",
        },
        inplace=True,
    )
    defendant_nodes.to_csv(graph_dir / "nodes_defendant.csv", index=False)

    article_nodes = articles_df[["article_label", "article_number"]].drop_duplicates().copy()
    if not article_nodes.empty:
        article_nodes.insert(0, "id:ID", article_nodes["article_label"].apply(lambda label: f"Article|{label}"))
        article_nodes.insert(1, ":LABEL", "Article")
        article_nodes.rename(columns={"article_label": "name"}, inplace=True)
        article_nodes.to_csv(graph_dir / "nodes_article.csv", index=False)

    guideline_nodes = guidelines_df[["guideline" ]].drop_duplicates().copy()
    if not guideline_nodes.empty:
        guideline_nodes.insert(0, "id:ID", guideline_nodes["guideline"].apply(lambda g: slugify(g, "Guideline", "GUIDELINE")))
        guideline_nodes.insert(1, ":LABEL", "Guideline")
        guideline_nodes.rename(columns={"guideline": "name"}, inplace=True)
        guideline_nodes.to_csv(graph_dir / "nodes_guideline.csv", index=False)

    context_nodes = contexts_df[["processing_context"]].drop_duplicates().copy()
    if not context_nodes.empty:
        context_nodes.insert(
            0,
            "id:ID",
            context_nodes["processing_context"].apply(lambda c: slugify(c, "Context", "CONTEXT")),
        )
        context_nodes.insert(1, ":LABEL", "Context")
        context_nodes.rename(columns={"processing_context": "name"}, inplace=True)
        context_nodes.to_csv(graph_dir / "nodes_context.csv", index=False)

    edges_decision_authority = enriched_df.dropna(subset=["authority_name_norm"])[["id", "authority_name_norm"]].copy()
    if not edges_decision_authority.empty:
        edges_decision_authority[":START_ID"] = edges_decision_authority["id"].apply(decision_node_id)
        edges_decision_authority[":END_ID"] = edges_decision_authority["authority_name_norm"].apply(
            lambda name: slugify(name, "Authority", "UNKNOWN")
        )
        edges_decision_authority[":TYPE"] = "ISSUED_BY"
        edges_decision_authority[[":START_ID", ":END_ID", ":TYPE"]].drop_duplicates().to_csv(
            graph_dir / "edges_decision_authority.csv", index=False
        )

    edges_decision_defendant = enriched_df.dropna(subset=["a7_defendant_name"])[["id", "a7_defendant_name"]].copy()
    if not edges_decision_defendant.empty:
        edges_decision_defendant[":START_ID"] = edges_decision_defendant["id"].apply(decision_node_id)
        edges_decision_defendant[":END_ID"] = edges_decision_defendant["a7_defendant_name"].apply(
            lambda name: slugify(name, "Defendant", "UNKNOWN_DEFENDANT")
        )
        edges_decision_defendant[":TYPE"] = "AGAINST"
        edges_decision_defendant[[":START_ID", ":END_ID", ":TYPE"]].drop_duplicates().to_csv(
            graph_dir / "edges_decision_defendant.csv", index=False
        )

    if not articles_df.empty:
        article_edges = articles_df.copy()
        article_edges[":START_ID"] = article_edges["id"].apply(decision_node_id)
        article_edges[":END_ID"] = article_edges["article_label"].apply(lambda label: f"Article|{label}")
        article_edges[":TYPE"] = "BREACHES"
        article_edges_out = article_edges[
            [
                ":START_ID",
                ":END_ID",
                ":TYPE",
                "article_reference",
                "article_reference_detail",
                "article_detail_tokens",
                "position",
            ]
        ].drop_duplicates()
        article_edges_out.to_csv(graph_dir / "edges_decision_article.csv", index=False)

    if not guidelines_df.empty:
        guideline_edges = guidelines_df.copy()
        guideline_edges[":START_ID"] = guideline_edges["id"].apply(decision_node_id)
        guideline_edges[":END_ID"] = guideline_edges["guideline"].apply(lambda g: slugify(g, "Guideline", "GUIDELINE"))
        guideline_edges[":TYPE"] = "REFERS_TO"
        guideline_edges[[":START_ID", ":END_ID", ":TYPE"]].drop_duplicates().to_csv(
            graph_dir / "edges_decision_guideline.csv", index=False
        )

    if not contexts_df.empty:
        context_edges = contexts_df.copy()
        context_edges[":START_ID"] = context_edges["id"].apply(decision_node_id)
        context_edges[":END_ID"] = context_edges["processing_context"].apply(
            lambda c: slugify(c, "Context", "CONTEXT")
        )
        context_edges[":TYPE"] = "INVOLVES"
        context_edges[[":START_ID", ":END_ID", ":TYPE"]].drop_duplicates().to_csv(
            graph_dir / "edges_decision_context.csv", index=False
        )


def enrich_dataset(
    input_path: Path,
    output_dir: Path,
    fx_path: Path,
    hicp_path: Path,
    context_taxonomy_path: Path,
    region_map_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fx_rates, hicp_index = load_reference_tables(fx_path, hicp_path)
    context_taxonomy = load_context_taxonomy(context_taxonomy_path)
    region_map = load_region_map(region_map_path)

    df = pd.read_csv(input_path)
    df = compute_temporal_features(df)
    df = convert_monetary_columns(df, fx_rates, hicp_index)

    contexts_df = explode_semicolon_list(df, "a14_processing_contexts", "processing_context")
    vulnerable_df = explode_semicolon_list(df, "a29_vulnerable_groups", "vulnerable_group")
    guidelines_df = explode_semicolon_list(df, "a74_guidelines_referenced", "guideline")
    articles_df = parse_articles(df)

    if not articles_df.empty:
        truncated_ids = set(articles_df.loc[articles_df["article_detail_truncated"], "id"].tolist())
    else:
        truncated_ids = set()
    df["flag_article_detail_truncated"] = df["id"].isin(truncated_ids)

    df = compute_art5_and_rights(df, articles_df)
    df = compute_measures(df)
    df = compute_art83_scores(df)
    df = compute_context_features(df, contexts_df, context_taxonomy)
    df = compute_complaint_and_flags(df)
    df = compute_oss_and_geography(df, region_map)
    df = compute_text_features(df, guidelines_df)
    df = compute_quality_flags(df)

    fx_metadata_columns = [
        "id",
        "fine_currency",
        "fine_amount_orig",
        "fine_amount_eur",
        "fine_fx_method",
        "fine_fx_year",
        "fine_fx_month",
        "flag_fine_fx_fallback",
        "turnover_currency",
        "turnover_amount_orig",
        "turnover_amount_eur",
        "turnover_fx_method",
        "turnover_fx_year",
        "turnover_fx_month",
        "flag_turnover_fx_fallback",
    ]

    fx_metadata = df[fx_metadata_columns].copy()
    fx_metadata.to_csv(output_dir / "0_fx_conversion_metadata.csv", index=False)

    fx_missing = df[
        (
            df["fine_amount_orig"].notna()
            & df["fine_amount_eur"].isna()
        )
        | (
            df["turnover_amount_orig"].notna()
            & df["turnover_amount_eur"].isna()
        )
    ][
        [
            "id",
            "fine_currency",
            "fine_amount_orig",
            "fine_fx_method",
            "turnover_currency",
            "turnover_amount_orig",
            "turnover_fx_method",
        ]
    ].copy()
    fx_missing.to_csv(output_dir / "0_fx_missing_review.csv", index=False)

    df.to_csv(output_dir / "1_enriched_master.csv", index=False)
    contexts_df.to_csv(output_dir / "2_processing_contexts.csv", index=False)
    vulnerable_df.to_csv(output_dir / "3_vulnerable_groups.csv", index=False)
    guidelines_df.to_csv(output_dir / "4_guidelines.csv", index=False)
    articles_df.to_csv(output_dir / "5_articles_breached.csv", index=False)

    build_graph_exports(df, contexts_df, guidelines_df, articles_df, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 enrichment pipeline")
    parser.add_argument(
        "--input",
        default="outputs/phase3_repair/repaired_dataset.csv",
        type=Path,
        help="Path to the Phase 3 repaired dataset",
    )
    parser.add_argument(
        "--output",
        default="outputs/phase4_enrichment",
        type=Path,
        help="Directory for enriched outputs",
    )
    parser.add_argument(
        "--fx-table",
        default="raw_data/reference/fx_rates.csv",
        type=Path,
        help="CSV containing FX rates to EUR",
    )
    parser.add_argument(
        "--hicp-table",
        default="raw_data/reference/hicp_ea19.csv",
        type=Path,
        help="CSV with HICP index for EUR deflation",
    )
    parser.add_argument(
        "--context-taxonomy",
        default="raw_data/reference/context_taxonomy.csv",
        type=Path,
        help="CSV describing processing context taxonomy and flag columns",
    )
    parser.add_argument(
        "--region-map",
        default="raw_data/reference/region_map.csv",
        type=Path,
        help="CSV mapping country codes to analyst region groupings",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enrich_dataset(
        args.input,
        args.output,
        args.fx_table,
        args.hicp_table,
        args.context_taxonomy,
        args.region_map,
    )
    print(f"✓ Phase 4 enrichment complete. Outputs written to {args.output}")


if __name__ == "__main__":
    main()
