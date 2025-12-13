"""Phase 1 – Paper Data Preparation: Analytical Sample and Indices.

This script implements Phase 1 (Data Preparation) of the methodology proposal:
"Reasoning and Consistency in GDPR Enforcement: A Cross-Border Analysis"

Tasks performed:
1. Load enriched master dataset from Phase 4
2. Construct analytical sample (fine-imposed filter)
3. Compute Authority Systematicity Index (Coverage × Consistency × Coherence)
4. Generate article cohort keys and membership tables
5. Prepare geographic and control variables

Input:  outputs/phase4_enrichment/1_enriched_master.csv
Output: outputs/paper/data/
        - analysis_sample.csv          (analytical sample for factor analysis)
        - authority_systematicity.csv  (authority-level systematicity index)
        - cohort_membership.csv        (article cohort membership)
        - sample_construction_log.txt  (sample flow documentation)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DATA_PATH = Path("outputs/phase4_enrichment/1_enriched_master.csv")
REGION_MAP_PATH = Path("raw_data/reference/region_map.csv")
OUTPUT_DIR = Path("outputs/paper/data")

# Random seed for reproducibility
RANDOM_SEED = 42

# Minimum decisions required for systematicity index
MIN_DECISIONS_SYSTEMATICITY = 10

# Number of Article 83(2) factors (a59-a69 = 11 factors)
NUM_ART83_FACTORS = 11

# Article 83(2) factor columns (from schema)
ART83_FACTOR_COLS = [
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

# Score columns (computed in Phase 4 enrichment)
ART83_SCORE_COLS = [f"{col}_score" for col in ART83_FACTOR_COLS]

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_master() -> pd.DataFrame:
    """Load the Phase 4 enriched master dataset."""
    if not BASE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Phase 4 master dataset not found at {BASE_DATA_PATH}. "
            "Run Phase 4 enrichment before paper analysis."
        )
    df = pd.read_csv(BASE_DATA_PATH, low_memory=False)
    logger.info(f"Loaded master dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_region_map() -> Dict[str, str]:
    """Load country-to-region mapping."""
    if not REGION_MAP_PATH.exists():
        logger.warning(f"Region map not found at {REGION_MAP_PATH}, using fallback")
        return {}
    region_df = pd.read_csv(REGION_MAP_PATH)
    return dict(zip(region_df["country_code"], region_df["region"]))


# -----------------------------------------------------------------------------
# Article Cohort Key Generation
# -----------------------------------------------------------------------------

ARTICLE_REGEX = re.compile(r"\d+")


def parse_articles(value: str) -> Tuple[int, ...]:
    """Parse article breached field into sorted tuple of integers."""
    if pd.isna(value):
        return tuple()
    text = str(value).strip()
    if text in {"", "NOT_DISCUSSED", "NOT_APPLICABLE", "NONE_VIOLATED"}:
        return tuple()
    numbers = {int(match.group(0)) for match in ARTICLE_REGEX.finditer(text)}
    return tuple(sorted(numbers))


def build_article_set_key(articles: Tuple[int, ...]) -> str:
    """Build deterministic key from article set."""
    if not articles:
        return "NONE"
    return ";".join(map(str, articles))


def build_article_family_key(articles: Tuple[int, ...]) -> str:
    """Map articles to families for relaxed matching."""
    ARTICLE_FAMILY_MAP: Dict[int, str] = {
        5: "5", 6: "6", 7: "7", 9: "9",
        12: "12-14", 13: "12-14", 14: "12-14",
        15: "15", 16: "16", 17: "17",
        21: "21", 22: "22",
        24: "24", 25: "25", 26: "26",
        28: "28", 29: "29",
        32: "32", 33: "33", 34: "34",
        35: "35", 37: "37", 44: "44",
        58: "58", 82: "82", 83: "83",
    }
    families = sorted({ARTICLE_FAMILY_MAP.get(num, str(num)) for num in articles})
    if not families:
        return "NONE"
    return ";".join(families)


# -----------------------------------------------------------------------------
# Systematicity Index Computation
# -----------------------------------------------------------------------------

@dataclass
class SystematicityComponents:
    """Container for systematicity index components."""
    authority_id: str
    n_decisions: int
    coverage: float      # Mean(discussed_count) / 11
    consistency: float   # 1 - CV(balance_score), bounded [0,1]
    coherence: float     # |cor(balance_score, log_fine)|
    systematicity: float # Coverage × Consistency × Coherence


def compute_authority_systematicity(
    df: pd.DataFrame,
    min_decisions: int = MIN_DECISIONS_SYSTEMATICITY
) -> List[SystematicityComponents]:
    """
    Compute systematicity index for each authority.

    Systematicity = Coverage × Consistency × Coherence

    Where:
    - Coverage = mean(art83_discussed_count) / 11
    - Consistency = 1 - CV(art83_balance_score), bounded [0, 1]
    - Coherence = |cor(art83_balance_score, log_fine_2025)|

    Only authorities with >= min_decisions are included.
    """
    # Ensure we have the required columns
    required_cols = ["authority_name_norm", "art83_discussed_count",
                     "art83_balance_score", "log_fine_2025"]

    # Use authority_name_norm if available, else fall back to a2_authority_name
    if "authority_name_norm" not in df.columns:
        df = df.copy()
        df["authority_name_norm"] = df["a2_authority_name"]

    # Compute log_fine_2025 if not present
    if "log_fine_2025" not in df.columns:
        df = df.copy()
        fine_col = "fine_amount_eur_real_2025"
        if fine_col not in df.columns:
            fine_col = "fine_amount_eur"
        df["log_fine_2025"] = np.log1p(df[fine_col].clip(lower=0).fillna(0))

    results: List[SystematicityComponents] = []

    for authority, group in df.groupby("authority_name_norm"):
        if pd.isna(authority) or str(authority).strip() == "":
            continue

        n_decisions = len(group)
        if n_decisions < min_decisions:
            continue

        # Filter to fine-imposed cases for coherence calculation
        fine_group = group[group["fine_imposed_bool"] == True].copy()

        # Coverage: mean discussed count / 11
        discussed_counts = group["art83_discussed_count"].dropna()
        if len(discussed_counts) == 0:
            coverage = 0.0
        else:
            coverage = discussed_counts.mean() / NUM_ART83_FACTORS

        # Consistency: 1 - normalized_std(balance_score)
        # Using normalized std relative to max possible range (22, from -11 to +11)
        # This avoids the CV problem when mean is near zero
        balance_scores = group["art83_balance_score"].dropna()
        MAX_BALANCE_RANGE = 2 * NUM_ART83_FACTORS  # 22 (from -11 to +11)
        if len(balance_scores) < 2:
            consistency = 0.0
        else:
            std_balance = balance_scores.std(ddof=1)
            # Normalize std by maximum possible std (half the range)
            normalized_std = std_balance / (MAX_BALANCE_RANGE / 2)
            consistency = max(0, min(1, 1 - normalized_std))

        # Coherence: |cor(balance_score, log_fine)|
        if len(fine_group) < 3:
            coherence = 0.0
        else:
            balance = fine_group["art83_balance_score"].dropna()
            log_fine = fine_group["log_fine_2025"].dropna()
            # Align indices
            common_idx = balance.index.intersection(log_fine.index)
            if len(common_idx) < 3:
                coherence = 0.0
            else:
                corr, _ = stats.pearsonr(balance.loc[common_idx], log_fine.loc[common_idx])
                coherence = abs(corr) if not np.isnan(corr) else 0.0

        # Systematicity = Coverage × Consistency × Coherence
        systematicity = coverage * consistency * coherence

        results.append(SystematicityComponents(
            authority_id=str(authority),
            n_decisions=n_decisions,
            coverage=coverage,
            consistency=consistency,
            coherence=coherence,
            systematicity=systematicity
        ))

    logger.info(f"Computed systematicity for {len(results)} authorities "
                f"(min {min_decisions} decisions)")
    return results


# -----------------------------------------------------------------------------
# Sample Construction
# -----------------------------------------------------------------------------

@dataclass
class SampleStats:
    """Container for sample construction statistics."""
    raw_count: int
    validated_count: int
    fine_imposed_count: int
    fine_positive_count: int
    analytical_sample_count: int
    cross_border_eligible: int
    exclusion_reasons: Dict[str, int]


def construct_analytical_sample(df: pd.DataFrame) -> Tuple[pd.DataFrame, SampleStats]:
    """
    Construct the analytical sample following methodology inclusion criteria.

    Inclusion for Factor Analysis (RQ1-2):
    - a53_fine_imposed = YES
    - fine_amount_eur > 0

    Returns:
        Tuple of (analytical_sample_df, sample_stats)
    """
    exclusions: Dict[str, int] = {}

    raw_count = len(df)
    logger.info(f"Starting sample construction from {raw_count} records")

    # Create working copy
    sample = df.copy()

    # Ensure log_fine_2025 exists
    if "log_fine_2025" not in sample.columns:
        fine_col = "fine_amount_eur_real_2025"
        if fine_col not in sample.columns:
            fine_col = "fine_amount_eur"
        sample["log_fine_2025"] = np.log1p(sample[fine_col].clip(lower=0).fillna(0))

    # Track validated records (non-null country code as proxy for valid parsing)
    valid_mask = sample["a1_country_code"].notna() & (sample["a1_country_code"] != "NOT_APPLICABLE")
    exclusions["invalid_record"] = (~valid_mask).sum()
    validated_count = valid_mask.sum()

    # Fine imposed filter
    fine_imposed_mask = sample["a53_fine_imposed"] == "YES"
    if "fine_imposed_bool" in sample.columns:
        fine_imposed_mask = fine_imposed_mask | (sample["fine_imposed_bool"] == True)
    exclusions["no_fine_imposed"] = (valid_mask & ~fine_imposed_mask).sum()
    fine_imposed_count = (valid_mask & fine_imposed_mask).sum()

    # Positive fine amount filter
    fine_col = "fine_amount_eur_real_2025" if "fine_amount_eur_real_2025" in sample.columns else "fine_amount_eur"
    fine_positive_mask = sample[fine_col].fillna(0) > 0
    exclusions["zero_or_missing_fine"] = (valid_mask & fine_imposed_mask & ~fine_positive_mask).sum()
    fine_positive_count = (valid_mask & fine_imposed_mask & fine_positive_mask).sum()

    # Analytical sample: valid + fine imposed + positive fine
    analytical_mask = valid_mask & fine_imposed_mask & fine_positive_mask
    analytical_sample = sample[analytical_mask].copy()
    analytical_sample_count = len(analytical_sample)

    # Parse articles and create cohort keys
    analytical_sample["article_set"] = analytical_sample["a77_articles_breached"].apply(parse_articles)
    analytical_sample["article_set_key"] = analytical_sample["article_set"].apply(build_article_set_key)
    analytical_sample["article_family_key"] = analytical_sample["article_set"].apply(build_article_family_key)
    analytical_sample["article_count"] = analytical_sample["article_set"].apply(len)

    # Cross-border eligibility: article cohort present in ≥2 countries
    cohort_country_counts = analytical_sample.groupby("article_set_key")["a1_country_code"].nunique()
    cross_border_cohorts = set(cohort_country_counts[cohort_country_counts >= 2].index)
    analytical_sample["cross_border_eligible"] = analytical_sample["article_set_key"].isin(cross_border_cohorts)
    cross_border_eligible = analytical_sample["cross_border_eligible"].sum()

    # Add region mapping
    region_map = load_region_map()
    analytical_sample["region"] = analytical_sample["a1_country_code"].map(region_map).fillna("Unknown")

    stats = SampleStats(
        raw_count=raw_count,
        validated_count=validated_count,
        fine_imposed_count=fine_imposed_count,
        fine_positive_count=fine_positive_count,
        analytical_sample_count=analytical_sample_count,
        cross_border_eligible=cross_border_eligible,
        exclusion_reasons=exclusions
    )

    logger.info(f"Analytical sample: {analytical_sample_count} records "
                f"({cross_border_eligible} cross-border eligible)")

    return analytical_sample, stats


def build_cohort_membership(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build cohort membership table showing article set distribution.

    Returns DataFrame with:
    - article_set_key
    - article_family_key
    - case_count
    - country_count
    - countries (semicolon-separated)
    - mean_log_fine
    - std_log_fine
    """
    records: List[Dict] = []

    for key, group in df.groupby("article_set_key"):
        countries = sorted(group["a1_country_code"].dropna().unique())
        records.append({
            "article_set_key": key,
            "article_family_key": group["article_family_key"].iloc[0],
            "article_count": group["article_count"].iloc[0],
            "case_count": len(group),
            "country_count": len(countries),
            "countries": ";".join(countries),
            "mean_log_fine": group["log_fine_2025"].mean(),
            "std_log_fine": group["log_fine_2025"].std(ddof=0),
            "median_fine_eur": group["fine_amount_eur_real_2025"].median()
                if "fine_amount_eur_real_2025" in group.columns
                else group["fine_amount_eur"].median(),
            "cross_border_eligible": len(countries) >= 2,
        })

    cohort_df = pd.DataFrame.from_records(records)
    cohort_df = cohort_df.sort_values("case_count", ascending=False)

    logger.info(f"Built cohort membership: {len(cohort_df)} unique article sets")
    return cohort_df


# -----------------------------------------------------------------------------
# Output Generation
# -----------------------------------------------------------------------------

def ensure_output_dir() -> None:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")


def write_sample_log(stats: SampleStats, systematicity_results: List[SystematicityComponents]) -> None:
    """Write sample construction log for transparency."""
    log_path = OUTPUT_DIR / "sample_construction_log.txt"

    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("GDPR ENFORCEMENT PAPER - SAMPLE CONSTRUCTION LOG\n")
        f.write("Phase 1: Data Preparation\n")
        f.write("=" * 70 + "\n\n")

        f.write("SAMPLE FLOW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Raw AI Responses (Phase 1):        n = {stats.raw_count:,}\n")
        f.write(f"Validated Records (Phase 2-3):     n = {stats.validated_count:,}\n")
        f.write(f"Fine Imposed (a53=YES):            n = {stats.fine_imposed_count:,}\n")
        f.write(f"Positive Fine Amount:              n = {stats.fine_positive_count:,}\n")
        f.write(f"Analytical Sample:                 n = {stats.analytical_sample_count:,}\n")
        f.write(f"Cross-Border Eligible (≥2 countries): n = {stats.cross_border_eligible:,}\n")
        f.write("\n")

        f.write("EXCLUSION BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for reason, count in stats.exclusion_reasons.items():
            f.write(f"{reason:30s}: {count:,}\n")
        f.write("\n")

        f.write("SYSTEMATICITY INDEX SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Authorities with ≥10 decisions: {len(systematicity_results)}\n")

        if systematicity_results:
            sys_values = [r.systematicity for r in systematicity_results]
            f.write(f"Systematicity Mean: {np.mean(sys_values):.4f}\n")
            f.write(f"Systematicity Median: {np.median(sys_values):.4f}\n")
            f.write(f"Systematicity Std: {np.std(sys_values):.4f}\n")
            f.write(f"Systematicity Range: [{min(sys_values):.4f}, {max(sys_values):.4f}]\n")

            # Top 5 authorities
            sorted_results = sorted(systematicity_results, key=lambda x: x.systematicity, reverse=True)
            f.write("\nTop 5 Authorities by Systematicity:\n")
            for i, r in enumerate(sorted_results[:5], 1):
                f.write(f"  {i}. {r.authority_id[:40]:40s} "
                        f"(n={r.n_decisions:3d}, sys={r.systematicity:.4f})\n")

            f.write("\nBottom 5 Authorities by Systematicity:\n")
            for i, r in enumerate(sorted_results[-5:], 1):
                f.write(f"  {i}. {r.authority_id[:40]:40s} "
                        f"(n={r.n_decisions:3d}, sys={r.systematicity:.4f})\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Generated by 6_paper_data_preparation.py\n")

    logger.info(f"Wrote sample construction log to {log_path}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main() -> None:
    """Execute Phase 1: Data Preparation."""
    logger.info("=" * 70)
    logger.info("PHASE 1: PAPER DATA PREPARATION")
    logger.info("=" * 70)

    # Set random seed
    np.random.seed(RANDOM_SEED)

    # Ensure output directory exists
    ensure_output_dir()

    # Load data
    logger.info("\n[1/5] Loading master dataset...")
    df = load_master()

    # Construct analytical sample
    logger.info("\n[2/5] Constructing analytical sample...")
    analytical_sample, sample_stats = construct_analytical_sample(df)

    # Compute systematicity indices
    logger.info("\n[3/5] Computing authority systematicity indices...")
    systematicity_results = compute_authority_systematicity(df)

    # Build cohort membership
    logger.info("\n[4/5] Building cohort membership table...")
    cohort_df = build_cohort_membership(analytical_sample)

    # Write outputs
    logger.info("\n[5/5] Writing outputs...")

    # 1. Analytical sample
    sample_path = OUTPUT_DIR / "analysis_sample.csv"
    # Drop the article_set column (contains tuples, not CSV-friendly)
    output_cols = [c for c in analytical_sample.columns if c != "article_set"]
    analytical_sample[output_cols].to_csv(sample_path, index=False)
    logger.info(f"  - analysis_sample.csv: {len(analytical_sample)} rows")

    # 2. Authority systematicity
    sys_df = pd.DataFrame([
        {
            "authority_id": r.authority_id,
            "n_decisions": r.n_decisions,
            "coverage": r.coverage,
            "consistency": r.consistency,
            "coherence": r.coherence,
            "systematicity": r.systematicity,
        }
        for r in systematicity_results
    ])
    sys_df = sys_df.sort_values("systematicity", ascending=False)
    sys_path = OUTPUT_DIR / "authority_systematicity.csv"
    sys_df.to_csv(sys_path, index=False)
    logger.info(f"  - authority_systematicity.csv: {len(sys_df)} authorities")

    # 3. Cohort membership
    cohort_path = OUTPUT_DIR / "cohort_membership.csv"
    cohort_df.to_csv(cohort_path, index=False)
    logger.info(f"  - cohort_membership.csv: {len(cohort_df)} cohorts")

    # 4. Sample construction log
    write_sample_log(sample_stats, systematicity_results)

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 COMPLETE")
    logger.info(f"Outputs written to: {OUTPUT_DIR}")
    logger.info("=" * 70)

    # Print summary
    print("\n" + "=" * 50)
    print("PHASE 1 SUMMARY")
    print("=" * 50)
    print(f"Analytical Sample:        {sample_stats.analytical_sample_count:,} decisions")
    print(f"Cross-Border Eligible:    {sample_stats.cross_border_eligible:,} decisions")
    print(f"Unique Article Cohorts:   {len(cohort_df):,}")
    print(f"Systematicity Authorities: {len(systematicity_results):,}")
    print("=" * 50)


if __name__ == "__main__":
    main()
