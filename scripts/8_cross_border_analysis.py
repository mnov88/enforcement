"""Phase 4 – Cross-Border Analysis: Matching and Variance Decomposition.

This script implements Phase 4 (Cross-Border Analysis) of the methodology proposal:
"Reasoning and Consistency in GDPR Enforcement: A Cross-Border Analysis"

Models and Analyses Implemented:
1. Model 4: Cross-border nearest-neighbor matching within article cohorts
2. Model 5: Three-level variance decomposition (country/authority/case)
3. Matched-pair gap statistics and hypothesis testing
4. Cross-country disparity quantification

Research Questions Addressed:
- RQ3: Do similar violations receive similar penalties across EU member states?
- RQ4: How much enforcement variance is attributable to authority vs. country vs. case?

Outputs:
- Table 6: Cross-border matched pairs mean gaps by article cohort
- Table 7: Variance decomposition: country vs. authority vs. case
- Figure 4: Cross-border fine gaps: matched-pair distributions
- Figure 5: Variance partition visualization

Input:  outputs/paper/data/analysis_sample.csv
Output: outputs/paper/tables/, outputs/paper/figures/
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Optional imports with fallbacks
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Variance decomposition will be limited.")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not installed. Figures will be skipped.")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ANALYSIS_SAMPLE_PATH = Path("outputs/paper/data/analysis_sample.csv")
COHORT_PATH = Path("outputs/paper/data/cohort_membership.csv")

TABLES_DIR = Path("outputs/paper/tables")
FIGURES_DIR = Path("outputs/paper/figures")
DATA_DIR = Path("outputs/paper/data")

RANDOM_SEED = 42

# Minimum cases per cohort for matching
MIN_COHORT_SIZE = 2
MIN_COUNTRIES_FOR_MATCHING = 2

# Mega-fine threshold for sensitivity analysis
MEGA_FINE_THRESHOLD = 10_000_000  # EUR 10 million

# Output directory for supplementary tables
SUPPLEMENTARY_DIR = Path("outputs/paper/supplementary")

# Matching variables and their types
MATCHING_VARS = {
    "defendant_class_encoded": "categorical",
    "enterprise_size_encoded": "categorical",
    "sector_encoded": "categorical",
    "decision_year": "continuous",
}

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Loading and Preparation
# -----------------------------------------------------------------------------

def load_analysis_sample() -> pd.DataFrame:
    """Load the analytical sample from Phase 1."""
    if not ANALYSIS_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Analysis sample not found at {ANALYSIS_SAMPLE_PATH}. "
            "Run 6_paper_data_preparation.py first."
        )
    df = pd.read_csv(ANALYSIS_SAMPLE_PATH, low_memory=False)
    logger.info(f"Loaded analysis sample: {len(df)} rows")
    return df


def prepare_matching_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for cross-border matching.

    Encodes categorical variables and ensures required columns exist.
    """
    data = df.copy()

    # Ensure log_fine_2025 exists
    if "log_fine_2025" not in data.columns:
        fine_col = "fine_amount_eur_real_2025" if "fine_amount_eur_real_2025" in data.columns else "fine_amount_eur"
        data["log_fine_2025"] = np.log1p(data[fine_col].clip(lower=0).fillna(0))

    # Encode defendant class
    class_map = {
        "PRIVATE": 1, "PUBLIC": 2, "NGO_OR_NONPROFIT": 3, "RELIGIOUS": 4,
        "INDIVIDUAL": 5, "POLITICAL_PARTY": 6, "OTHER": 7
    }
    data["defendant_class_encoded"] = data["a8_defendant_class"].map(class_map).fillna(7)

    # Encode enterprise size
    size_map = {
        "SME": 1, "LARGE": 2, "VERY_LARGE": 3, "UNKNOWN": 4, "NOT_APPLICABLE": 4
    }
    data["enterprise_size_encoded"] = data["a9_enterprise_size"].map(size_map).fillna(4)

    # Encode sector (simplified grouping)
    sector_groups = {
        "FINANCIAL_SERVICES": 1, "HEALTH_CARE": 2, "TELECOMMUNICATIONS": 3,
        "TECHNOLOGY": 4, "MEDIA_ENTERTAINMENT": 5, "RETAIL": 6,
        "TRANSPORTATION": 7, "ENERGY_UTILITIES": 8, "EDUCATION": 9,
        "REAL_ESTATE": 10, "PUBLIC_ADMINISTRATION": 11, "OTHER": 12
    }
    data["sector_encoded"] = data["a12_sector"].map(sector_groups).fillna(12)

    # Ensure country code is clean
    data["country_code"] = data["a1_country_code"].fillna("UNKNOWN")

    # Ensure authority name is available
    if "authority_name_norm" not in data.columns:
        data["authority_name_norm"] = data["a2_authority_name"]

    # Ensure article_set_key exists
    if "article_set_key" not in data.columns:
        logger.warning("article_set_key not found, creating from a77_articles_breached")
        data["article_set_key"] = data["a77_articles_breached"].fillna("NONE")

    # Create unique case ID if not present
    if "case_id" not in data.columns:
        data["case_id"] = data.index.astype(str) if "id" not in data.columns else data["id"]
    else:
        data["case_id"] = data["case_id"].astype(str)

    # Get fine amount in EUR
    data["fine_eur"] = data["fine_amount_eur_real_2025"].fillna(
        data["fine_amount_eur"] if "fine_amount_eur" in data.columns else 0
    )

    logger.info(f"Prepared matching data: {len(data)} cases, {data['country_code'].nunique()} countries")
    return data


# -----------------------------------------------------------------------------
# Model 4: Cross-Border Nearest-Neighbor Matching
# -----------------------------------------------------------------------------

@dataclass
class MatchedPair:
    """Container for a single matched pair."""
    case_i_id: str
    case_j_id: str
    country_i: str
    country_j: str
    article_cohort: str
    log_fine_i: float
    log_fine_j: float
    fine_eur_i: float
    fine_eur_j: float
    delta_log_fine: float  # |log_fine_i - log_fine_j|
    delta_fine_eur: float  # |fine_eur_i - fine_eur_j|
    distance: float  # Mahalanobis or normalized distance
    defendant_class_match: bool
    enterprise_size_match: bool
    year_diff: int


@dataclass
class CohortMatchResults:
    """Container for matching results within a cohort."""
    article_cohort: str
    n_cases: int
    n_countries: int
    n_matched_pairs: int
    pairs: List[MatchedPair]
    mean_delta_log_fine: float
    std_delta_log_fine: float
    mean_delta_fine_eur: float
    median_delta_fine_eur: float


def compute_distance_matrix(
    df: pd.DataFrame,
    match_vars: List[str]
) -> np.ndarray:
    """
    Compute pairwise distance matrix for matching.

    Uses normalized Euclidean distance for mixed categorical/continuous variables.
    """
    # Prepare feature matrix
    X = df[match_vars].values.astype(float)

    # Standardize each column (z-score)
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    # Compute pairwise Euclidean distance
    dist_matrix = cdist(X_std, X_std, metric='euclidean')

    return dist_matrix


def find_cross_country_matches(
    cohort_df: pd.DataFrame,
    max_matches_per_case: int = 3
) -> List[MatchedPair]:
    """
    Find nearest-neighbor matches across different countries within a cohort.

    For each case i in country c, find the nearest case j in country c' != c.
    Uses greedy matching to avoid many-to-one matches.
    """
    match_vars = ["defendant_class_encoded", "enterprise_size_encoded",
                  "sector_encoded", "decision_year"]

    # Filter to complete cases
    match_cols = match_vars + ["country_code", "case_id", "log_fine_2025",
                               "fine_eur", "article_set_key", "a8_defendant_class",
                               "a9_enterprise_size"]
    complete_df = cohort_df[match_cols].dropna()

    if len(complete_df) < 2:
        return []

    countries = complete_df["country_code"].unique()
    if len(countries) < 2:
        return []

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(complete_df, match_vars)

    # Set diagonal and same-country distances to infinity
    n = len(complete_df)
    for i in range(n):
        dist_matrix[i, i] = np.inf
        for j in range(n):
            if complete_df.iloc[i]["country_code"] == complete_df.iloc[j]["country_code"]:
                dist_matrix[i, j] = np.inf

    pairs: List[MatchedPair] = []
    used_j = set()  # Track already-matched cases to avoid many-to-one

    # For each case, find best cross-country match
    for i in range(n):
        row_i = complete_df.iloc[i]

        # Find nearest unmatched case in different country
        available_mask = np.array([j not in used_j for j in range(n)])
        masked_distances = np.where(available_mask, dist_matrix[i], np.inf)

        best_j = np.argmin(masked_distances)
        best_dist = masked_distances[best_j]

        if best_dist == np.inf:
            continue  # No valid match found

        row_j = complete_df.iloc[best_j]

        # Create matched pair
        pair = MatchedPair(
            case_i_id=str(row_i["case_id"]),
            case_j_id=str(row_j["case_id"]),
            country_i=row_i["country_code"],
            country_j=row_j["country_code"],
            article_cohort=row_i["article_set_key"],
            log_fine_i=row_i["log_fine_2025"],
            log_fine_j=row_j["log_fine_2025"],
            fine_eur_i=row_i["fine_eur"],
            fine_eur_j=row_j["fine_eur"],
            delta_log_fine=abs(row_i["log_fine_2025"] - row_j["log_fine_2025"]),
            delta_fine_eur=abs(row_i["fine_eur"] - row_j["fine_eur"]),
            distance=best_dist,
            defendant_class_match=(row_i["a8_defendant_class"] == row_j["a8_defendant_class"]),
            enterprise_size_match=(row_i["a9_enterprise_size"] == row_j["a9_enterprise_size"]),
            year_diff=abs(int(row_i["decision_year"]) - int(row_j["decision_year"])),
        )
        pairs.append(pair)
        used_j.add(best_j)

    return pairs


def run_cross_border_matching(df: pd.DataFrame) -> List[CohortMatchResults]:
    """
    Run cross-border matching across all article cohorts.

    Returns matching results for each cohort with sufficient cross-country coverage.
    """
    results: List[CohortMatchResults] = []

    # Get cohorts with cross-border eligibility
    cohort_stats = df.groupby("article_set_key").agg({
        "country_code": "nunique",
        "case_id": "count"
    }).reset_index()
    cohort_stats.columns = ["article_set_key", "n_countries", "n_cases"]

    eligible_cohorts = cohort_stats[
        (cohort_stats["n_countries"] >= MIN_COUNTRIES_FOR_MATCHING) &
        (cohort_stats["n_cases"] >= MIN_COHORT_SIZE)
    ]["article_set_key"].tolist()

    logger.info(f"Found {len(eligible_cohorts)} cohorts eligible for cross-border matching")

    for cohort in eligible_cohorts:
        cohort_df = df[df["article_set_key"] == cohort].copy()

        pairs = find_cross_country_matches(cohort_df)

        if len(pairs) < 1:
            continue

        # Compute cohort-level statistics
        delta_logs = [p.delta_log_fine for p in pairs]
        delta_eurs = [p.delta_fine_eur for p in pairs]

        results.append(CohortMatchResults(
            article_cohort=cohort,
            n_cases=len(cohort_df),
            n_countries=cohort_df["country_code"].nunique(),
            n_matched_pairs=len(pairs),
            pairs=pairs,
            mean_delta_log_fine=np.mean(delta_logs),
            std_delta_log_fine=np.std(delta_logs, ddof=1) if len(delta_logs) > 1 else 0,
            mean_delta_fine_eur=np.mean(delta_eurs),
            median_delta_fine_eur=np.median(delta_eurs),
        ))

    total_pairs = sum(r.n_matched_pairs for r in results)
    logger.info(f"Completed matching: {len(results)} cohorts, {total_pairs} total pairs")

    return results


def test_cross_border_disparity(all_pairs: List[MatchedPair]) -> Dict[str, Any]:
    """
    Test whether cross-border disparity is significantly different from zero.

    H0: E[delta_log_fine] = 0 (no systematic cross-border disparity)
    H1: E[delta_log_fine] > 0 (systematic disparity)
    """
    if len(all_pairs) < 5:
        return {"test": "insufficient_data", "n_pairs": len(all_pairs)}

    delta_logs = np.array([p.delta_log_fine for p in all_pairs])

    # One-sample t-test against 0
    t_stat, p_value_two = stats.ttest_1samp(delta_logs, 0)
    p_value_one = p_value_two / 2 if t_stat > 0 else 1 - p_value_two / 2

    # Signed rank test (non-parametric)
    # For absolute differences, test if median is significantly > 0
    # Use Wilcoxon signed-rank on (delta - 0)
    try:
        w_stat, w_pval = stats.wilcoxon(delta_logs, alternative='greater')
    except Exception:
        w_stat, w_pval = np.nan, np.nan

    # Effect size: Cohen's d
    cohens_d = np.mean(delta_logs) / np.std(delta_logs, ddof=1) if np.std(delta_logs, ddof=1) > 0 else 0

    return {
        "test": "disparity_test",
        "n_pairs": len(all_pairs),
        "mean_delta_log": np.mean(delta_logs),
        "std_delta_log": np.std(delta_logs, ddof=1),
        "median_delta_log": np.median(delta_logs),
        "t_stat": t_stat,
        "p_value_onesided": p_value_one,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pval": w_pval,
        "cohens_d": cohens_d,
        "significant_005": p_value_one < 0.05,
    }


def format_table6_cross_border_gaps(
    cohort_results: List[CohortMatchResults],
    disparity_test: Dict[str, Any]
) -> pd.DataFrame:
    """Format Table 6: Cross-border matched pairs mean gaps by article cohort."""
    rows = []

    # Sort by number of matched pairs (descending)
    sorted_results = sorted(cohort_results, key=lambda x: x.n_matched_pairs, reverse=True)

    for r in sorted_results[:25]:  # Top 25 cohorts by pair count
        rows.append({
            "Article Cohort": r.article_cohort,
            "N Cases": r.n_cases,
            "N Countries": r.n_countries,
            "N Pairs": r.n_matched_pairs,
            "Mean Δ Log Fine": f"{r.mean_delta_log_fine:.3f}",
            "SD Δ Log Fine": f"{r.std_delta_log_fine:.3f}",
            "Mean Δ Fine (EUR)": f"{r.mean_delta_fine_eur:,.0f}",
            "Median Δ Fine (EUR)": f"{r.median_delta_fine_eur:,.0f}",
        })

    # Add summary row
    all_pairs = [p for r in cohort_results for p in r.pairs]
    rows.append({
        "Article Cohort": "--- TOTAL ---",
        "N Cases": sum(r.n_cases for r in cohort_results),
        "N Countries": "-",
        "N Pairs": len(all_pairs),
        "Mean Δ Log Fine": f"{disparity_test.get('mean_delta_log', 0):.3f}",
        "SD Δ Log Fine": f"{disparity_test.get('std_delta_log', 0):.3f}",
        "Mean Δ Fine (EUR)": f"{np.mean([p.delta_fine_eur for p in all_pairs]):,.0f}" if all_pairs else "-",
        "Median Δ Fine (EUR)": f"{np.median([p.delta_fine_eur for p in all_pairs]):,.0f}" if all_pairs else "-",
    })

    # Add test statistics row
    rows.append({
        "Article Cohort": "--- TEST ---",
        "N Cases": f"t = {disparity_test.get('t_stat', 0):.3f}",
        "N Countries": f"p = {disparity_test.get('p_value_onesided', 1):.4f}",
        "N Pairs": f"d = {disparity_test.get('cohens_d', 0):.3f}",
        "Mean Δ Log Fine": "",
        "SD Δ Log Fine": "",
        "Mean Δ Fine (EUR)": "",
        "Median Δ Fine (EUR)": "",
    })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Model 5: Three-Level Variance Decomposition
# -----------------------------------------------------------------------------

@dataclass
class VarianceComponents:
    """Container for variance decomposition results."""
    n_obs: int
    n_countries: int
    n_authorities: int
    var_country: float
    var_authority: float  # Within-country, between-authority
    var_residual: float  # Within-authority (case-level)
    var_total: float
    pct_country: float
    pct_authority: float
    pct_residual: float
    icc_country: float  # Intraclass correlation at country level
    icc_authority: float  # Intraclass correlation at authority level
    model_converged: bool
    model_notes: str


def run_variance_decomposition(df: pd.DataFrame) -> Optional[VarianceComponents]:
    """
    Run three-level variance decomposition.

    Model: log_fine ~ 1 + (1|country) + (1|authority:country)

    Partitions variance into:
    - Between-country variance
    - Between-authority (within-country) variance
    - Within-authority (residual) variance
    """
    if not HAS_STATSMODELS:
        logger.error("statsmodels required for variance decomposition")
        return None

    # Prepare data
    model_df = df[["log_fine_2025", "country_code", "authority_name_norm"]].dropna().copy()

    if len(model_df) < 50:
        logger.warning("Insufficient observations for variance decomposition")
        return None

    n_countries = model_df["country_code"].nunique()
    n_authorities = model_df["authority_name_norm"].nunique()

    logger.info(f"Variance decomposition: {len(model_df)} obs, {n_countries} countries, {n_authorities} authorities")

    # Method 1: Two separate mixed models to extract variance components
    # This is a practical approach when true nested random effects are complex

    try:
        # Model 1: Authority as random effect (captures both country and authority variance)
        model_auth = smf.mixedlm(
            "log_fine_2025 ~ 1",
            data=model_df,
            groups=model_df["authority_name_norm"],
            re_formula="~1"
        )
        fit_auth = model_auth.fit(method="powell", maxiter=500)

        var_auth_total = float(fit_auth.cov_re.iloc[0, 0]) if hasattr(fit_auth.cov_re, 'iloc') else float(fit_auth.cov_re)
        var_resid = fit_auth.scale

        # Model 2: Country as random effect
        model_country = smf.mixedlm(
            "log_fine_2025 ~ 1",
            data=model_df,
            groups=model_df["country_code"],
            re_formula="~1"
        )
        fit_country = model_country.fit(method="powell", maxiter=500)

        var_country = float(fit_country.cov_re.iloc[0, 0]) if hasattr(fit_country.cov_re, 'iloc') else float(fit_country.cov_re)

        # Approximate authority-within-country variance
        # var_auth_total ≈ var_country + var_authority_within
        var_authority = max(0, var_auth_total - var_country)

        # Total variance
        var_total = var_country + var_authority + var_resid

        # Percentages
        pct_country = var_country / var_total * 100 if var_total > 0 else 0
        pct_authority = var_authority / var_total * 100 if var_total > 0 else 0
        pct_residual = var_resid / var_total * 100 if var_total > 0 else 0

        # ICCs
        icc_country = var_country / var_total if var_total > 0 else 0
        icc_authority = (var_country + var_authority) / var_total if var_total > 0 else 0

        return VarianceComponents(
            n_obs=len(model_df),
            n_countries=n_countries,
            n_authorities=n_authorities,
            var_country=var_country,
            var_authority=var_authority,
            var_residual=var_resid,
            var_total=var_total,
            pct_country=pct_country,
            pct_authority=pct_authority,
            pct_residual=pct_residual,
            icc_country=icc_country,
            icc_authority=icc_authority,
            model_converged=True,
            model_notes="Two-model approximation for nested random effects"
        )

    except Exception as e:
        logger.error(f"Variance decomposition failed: {e}")

        # Fallback: Simple ANOVA-based decomposition
        try:
            grand_mean = model_df["log_fine_2025"].mean()

            # Between-country variance
            country_means = model_df.groupby("country_code")["log_fine_2025"].mean()
            ss_country = sum(
                len(model_df[model_df["country_code"] == c]) * (m - grand_mean)**2
                for c, m in country_means.items()
            )

            # Between-authority variance (within country)
            auth_means = model_df.groupby("authority_name_norm")["log_fine_2025"].mean()
            ss_auth = sum(
                len(model_df[model_df["authority_name_norm"] == a]) * (m - grand_mean)**2
                for a, m in auth_means.items()
            ) - ss_country
            ss_auth = max(0, ss_auth)

            # Within-authority variance
            ss_resid = sum((model_df["log_fine_2025"] - grand_mean)**2) - ss_country - ss_auth
            ss_total = ss_country + ss_auth + ss_resid

            return VarianceComponents(
                n_obs=len(model_df),
                n_countries=n_countries,
                n_authorities=n_authorities,
                var_country=ss_country / len(model_df),
                var_authority=ss_auth / len(model_df),
                var_residual=ss_resid / len(model_df),
                var_total=ss_total / len(model_df),
                pct_country=ss_country / ss_total * 100 if ss_total > 0 else 0,
                pct_authority=ss_auth / ss_total * 100 if ss_total > 0 else 0,
                pct_residual=ss_resid / ss_total * 100 if ss_total > 0 else 0,
                icc_country=ss_country / ss_total if ss_total > 0 else 0,
                icc_authority=(ss_country + ss_auth) / ss_total if ss_total > 0 else 0,
                model_converged=False,
                model_notes="ANOVA-based fallback (mixed model failed)"
            )
        except Exception as e2:
            logger.error(f"Fallback variance decomposition also failed: {e2}")
            return None


def format_table7_variance_decomposition(vc: VarianceComponents) -> pd.DataFrame:
    """Format Table 7: Variance decomposition results."""
    rows = [
        {
            "Component": "Between-Country",
            "Variance": f"{vc.var_country:.4f}",
            "% of Total": f"{vc.pct_country:.1f}%",
            "ICC": f"{vc.icc_country:.4f}",
            "Interpretation": "National policy/legal tradition differences"
        },
        {
            "Component": "Between-Authority (within-country)",
            "Variance": f"{vc.var_authority:.4f}",
            "% of Total": f"{vc.pct_authority:.1f}%",
            "ICC": "-",
            "Interpretation": "\"Which DPA you get\" effect"
        },
        {
            "Component": "Within-Authority (case-level)",
            "Variance": f"{vc.var_residual:.4f}",
            "% of Total": f"{vc.pct_residual:.1f}%",
            "ICC": "-",
            "Interpretation": "Case-specific factors, measurement error"
        },
        {
            "Component": "---",
            "Variance": "---",
            "% of Total": "---",
            "ICC": "---",
            "Interpretation": "---"
        },
        {
            "Component": "Total",
            "Variance": f"{vc.var_total:.4f}",
            "% of Total": "100.0%",
            "ICC": f"{vc.icc_authority:.4f}",
            "Interpretation": f"N={vc.n_obs}, {vc.n_countries} countries, {vc.n_authorities} authorities"
        },
        {
            "Component": "Model Note",
            "Variance": "",
            "% of Total": "",
            "ICC": "",
            "Interpretation": vc.model_notes
        },
    ]

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Mega-Fine Sensitivity Analysis (NEW)
# -----------------------------------------------------------------------------

@dataclass
class MegaFineSensitivityResult:
    """Container for mega-fine sensitivity comparison."""
    analysis_type: str
    sample_description: str
    metric_name: str
    full_sample_value: float
    restricted_value: float
    pct_change: float
    full_n: int
    restricted_n: int
    notes: str


def run_cross_border_mega_fine_sensitivity(
    cohort_results: List[CohortMatchResults],
    disparity_test_full: Dict[str, Any],
    df: pd.DataFrame
) -> List[MegaFineSensitivityResult]:
    """
    Run sensitivity analysis excluding mega-fine pairs from cross-border analysis.

    Tests whether extreme fines (>EUR 10M) drive the cross-border disparity findings.
    """
    results: List[MegaFineSensitivityResult] = []

    # Get all pairs
    all_pairs = [p for r in cohort_results for p in r.pairs]

    if len(all_pairs) < 10:
        return results

    # Filter pairs: exclude if either case has mega-fine
    non_mega_pairs = [
        p for p in all_pairs
        if p.fine_eur_i < MEGA_FINE_THRESHOLD and p.fine_eur_j < MEGA_FINE_THRESHOLD
    ]

    # Also test with €1M threshold
    under_1m_pairs = [
        p for p in all_pairs
        if p.fine_eur_i < 1_000_000 and p.fine_eur_j < 1_000_000
    ]

    # Run disparity test on filtered samples
    def calc_disparity_stats(pairs):
        if len(pairs) < 5:
            return {"mean_delta_log": np.nan, "t_stat": np.nan, "p_value": np.nan, "cohens_d": np.nan}
        delta_logs = np.array([p.delta_log_fine for p in pairs])
        mean_d = np.mean(delta_logs)
        std_d = np.std(delta_logs, ddof=1)
        t_stat, p_two = stats.ttest_1samp(delta_logs, 0)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        cohens_d = mean_d / std_d if std_d > 0 else 0
        return {"mean_delta_log": mean_d, "t_stat": t_stat, "p_value": p_one, "cohens_d": cohens_d}

    full_stats = disparity_test_full
    non_mega_stats = calc_disparity_stats(non_mega_pairs)
    under_1m_stats = calc_disparity_stats(under_1m_pairs)

    # Compare mean delta log fine
    full_mean = full_stats.get('mean_delta_log', 0)
    non_mega_mean = non_mega_stats['mean_delta_log']
    under_1m_mean = under_1m_stats['mean_delta_log']

    results.append(MegaFineSensitivityResult(
        analysis_type="Cross-Border Disparity",
        sample_description="Exclude >€10M pairs",
        metric_name="Mean Δ log fine",
        full_sample_value=full_mean,
        restricted_value=non_mega_mean,
        pct_change=100 * (non_mega_mean - full_mean) / abs(full_mean) if full_mean else 0,
        full_n=len(all_pairs),
        restricted_n=len(non_mega_pairs),
        notes=f"Pairs excluded: {len(all_pairs) - len(non_mega_pairs)}"
    ))

    results.append(MegaFineSensitivityResult(
        analysis_type="Cross-Border Disparity",
        sample_description="Exclude >€1M pairs",
        metric_name="Mean Δ log fine",
        full_sample_value=full_mean,
        restricted_value=under_1m_mean,
        pct_change=100 * (under_1m_mean - full_mean) / abs(full_mean) if full_mean else 0,
        full_n=len(all_pairs),
        restricted_n=len(under_1m_pairs),
        notes=f"Pairs excluded: {len(all_pairs) - len(under_1m_pairs)}"
    ))

    # Compare Cohen's d
    full_d = full_stats.get('cohens_d', 0)
    non_mega_d = non_mega_stats['cohens_d']
    under_1m_d = under_1m_stats['cohens_d']

    results.append(MegaFineSensitivityResult(
        analysis_type="Cross-Border Disparity",
        sample_description="Exclude >€10M pairs",
        metric_name="Cohen's d",
        full_sample_value=full_d,
        restricted_value=non_mega_d,
        pct_change=100 * (non_mega_d - full_d) / abs(full_d) if full_d else 0,
        full_n=len(all_pairs),
        restricted_n=len(non_mega_pairs),
        notes="Effect size"
    ))

    # Implied fine ratio (exp of mean delta log)
    full_ratio = np.exp(full_mean)
    non_mega_ratio = np.exp(non_mega_mean) if not np.isnan(non_mega_mean) else np.nan
    under_1m_ratio = np.exp(under_1m_mean) if not np.isnan(under_1m_mean) else np.nan

    results.append(MegaFineSensitivityResult(
        analysis_type="Cross-Border Disparity",
        sample_description="Exclude >€10M pairs",
        metric_name="Implied fine ratio (exp)",
        full_sample_value=full_ratio,
        restricted_value=non_mega_ratio,
        pct_change=100 * (non_mega_ratio - full_ratio) / full_ratio if full_ratio and not np.isnan(non_mega_ratio) else 0,
        full_n=len(all_pairs),
        restricted_n=len(non_mega_pairs),
        notes=f"~{full_ratio:.1f}x vs ~{non_mega_ratio:.1f}x average gap"
    ))

    results.append(MegaFineSensitivityResult(
        analysis_type="Cross-Border Disparity",
        sample_description="Exclude >€1M pairs",
        metric_name="Implied fine ratio (exp)",
        full_sample_value=full_ratio,
        restricted_value=under_1m_ratio,
        pct_change=100 * (under_1m_ratio - full_ratio) / full_ratio if full_ratio and not np.isnan(under_1m_ratio) else 0,
        full_n=len(all_pairs),
        restricted_n=len(under_1m_pairs),
        notes=f"~{full_ratio:.1f}x vs ~{under_1m_ratio:.1f}x average gap"
    ))

    # Variance decomposition sensitivity
    if HAS_STATSMODELS:
        # Run variance decomposition excluding mega-fines
        non_mega_df = df[df["fine_eur"] < MEGA_FINE_THRESHOLD].copy()
        under_1m_df = df[df["fine_eur"] < 1_000_000].copy()

        vc_full = run_variance_decomposition(df)
        vc_non_mega = run_variance_decomposition(non_mega_df) if len(non_mega_df) >= 50 else None
        vc_under_1m = run_variance_decomposition(under_1m_df) if len(under_1m_df) >= 50 else None

        if vc_full and vc_non_mega:
            results.append(MegaFineSensitivityResult(
                analysis_type="Variance Decomposition",
                sample_description="Exclude >€10M fines",
                metric_name="Authority variance %",
                full_sample_value=vc_full.pct_authority,
                restricted_value=vc_non_mega.pct_authority,
                pct_change=vc_non_mega.pct_authority - vc_full.pct_authority,
                full_n=vc_full.n_obs,
                restricted_n=vc_non_mega.n_obs,
                notes="Percentage point change"
            ))

            results.append(MegaFineSensitivityResult(
                analysis_type="Variance Decomposition",
                sample_description="Exclude >€10M fines",
                metric_name="ICC (authority)",
                full_sample_value=vc_full.icc_authority,
                restricted_value=vc_non_mega.icc_authority,
                pct_change=100 * (vc_non_mega.icc_authority - vc_full.icc_authority) / vc_full.icc_authority if vc_full.icc_authority else 0,
                full_n=vc_full.n_obs,
                restricted_n=vc_non_mega.n_obs,
                notes="Combined country + authority ICC"
            ))

        if vc_full and vc_under_1m:
            results.append(MegaFineSensitivityResult(
                analysis_type="Variance Decomposition",
                sample_description="Exclude >€1M fines",
                metric_name="Authority variance %",
                full_sample_value=vc_full.pct_authority,
                restricted_value=vc_under_1m.pct_authority,
                pct_change=vc_under_1m.pct_authority - vc_full.pct_authority,
                full_n=vc_full.n_obs,
                restricted_n=vc_under_1m.n_obs,
                notes="Percentage point change"
            ))

    return results


def format_table_mega_fine_sensitivity(results: List[MegaFineSensitivityResult]) -> pd.DataFrame:
    """Format mega-fine sensitivity results as Table S7."""
    rows = []

    for r in results:
        rows.append({
            "Analysis": r.analysis_type,
            "Sample": r.sample_description,
            "Metric": r.metric_name,
            "Full Sample": f"{r.full_sample_value:.3f}" if not np.isnan(r.full_sample_value) else "—",
            "Restricted": f"{r.restricted_value:.3f}" if not np.isnan(r.restricted_value) else "—",
            "% Change": f"{r.pct_change:+.1f}%" if not np.isnan(r.pct_change) else "—",
            "N (Full)": r.full_n,
            "N (Restr.)": r.restricted_n,
            "Notes": r.notes,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def create_figure4_matched_pairs(
    cohort_results: List[CohortMatchResults],
    disparity_test: Dict[str, Any]
) -> Optional[plt.Figure]:
    """Create Figure 4: Cross-border fine gaps matched-pair distributions."""
    if not HAS_PLOTTING:
        return None

    all_pairs = [p for r in cohort_results for p in r.pairs]

    if len(all_pairs) < 5:
        logger.warning("Insufficient pairs for Figure 4")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Distribution of log fine gaps
    delta_logs = [p.delta_log_fine for p in all_pairs]

    ax1 = axes[0]
    ax1.hist(delta_logs, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(delta_logs), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {np.mean(delta_logs):.3f}')
    ax1.axvline(np.median(delta_logs), color='orange', linestyle=':', linewidth=2,
                label=f'Median = {np.median(delta_logs):.3f}')
    ax1.set_xlabel('Absolute Difference in Log Fine (|Δ log fine|)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('A. Distribution of Cross-Border Fine Gaps\n(Matched Pairs)', fontsize=12)
    ax1.legend(loc='upper right')

    # Add test result annotation
    t_stat = disparity_test.get('t_stat', 0)
    p_val = disparity_test.get('p_value_onesided', 1)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    ax1.annotate(
        f"t = {t_stat:.2f}, p = {p_val:.4f}{sig}\nN = {len(all_pairs)} pairs",
        xy=(0.95, 0.95), xycoords='axes fraction',
        ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Right panel: Fine gaps by cohort size
    ax2 = axes[1]

    # Prepare data: cohort mean gap vs cohort size
    cohort_data = [(r.n_cases, r.mean_delta_log_fine, r.n_matched_pairs)
                   for r in cohort_results if r.n_matched_pairs >= 2]

    if cohort_data:
        sizes, gaps, n_pairs = zip(*cohort_data)
        scatter = ax2.scatter(sizes, gaps, s=[n*10 for n in n_pairs],
                              alpha=0.6, c=n_pairs, cmap='viridis', edgecolor='black')
        ax2.set_xlabel('Cohort Size (N cases)', fontsize=11)
        ax2.set_ylabel('Mean Δ Log Fine', fontsize=11)
        ax2.set_title('B. Cross-Border Gaps by Article Cohort\n(bubble size = N pairs)', fontsize=12)
        plt.colorbar(scatter, ax=ax2, label='N Pairs')

        # Add trend line
        if len(sizes) >= 5:
            z = np.polyfit(sizes, gaps, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(sizes), max(sizes), 100)
            ax2.plot(x_line, p(x_line), 'r--', alpha=0.7, label='Trend')
            ax2.legend(loc='upper right')

    plt.tight_layout()
    return fig


def create_figure5_variance_partition(vc: VarianceComponents) -> Optional[plt.Figure]:
    """Create Figure 5: Variance partition pie chart."""
    if not HAS_PLOTTING:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Pie chart
    ax1 = axes[0]
    labels = ['Country', 'Authority\n(within-country)', 'Case\n(residual)']
    sizes = [vc.pct_country, vc.pct_authority, vc.pct_residual]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    explode = (0.02, 0.02, 0.02)

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.6,
        textprops={'fontsize': 11}
    )
    ax1.set_title('A. Variance Partition in Log Fine\n(Three-Level Model)', fontsize=12)

    # Right panel: Bar chart with annotations
    ax2 = axes[1]
    components = ['Country\n(σ²_c)', 'Authority\n(σ²_a)', 'Residual\n(σ²_e)']
    variances = [vc.var_country, vc.var_authority, vc.var_residual]
    percentages = [vc.pct_country, vc.pct_authority, vc.pct_residual]

    bars = ax2.bar(components, variances, color=colors, edgecolor='black', alpha=0.8)

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax2.annotate(f'{pct:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Variance', fontsize=11)
    ax2.set_title('B. Variance Components\n(Log Fine)', fontsize=12)
    ax2.set_ylim(0, max(variances) * 1.2)

    # Add summary annotation
    summary_text = (
        f"ICC (Country): {vc.icc_country:.3f}\n"
        f"ICC (Country + Authority): {vc.icc_authority:.3f}\n"
        f"N = {vc.n_obs} decisions\n"
        f"{vc.n_countries} countries, {vc.n_authorities} authorities"
    )
    ax2.annotate(summary_text, xy=(0.98, 0.98), xycoords='axes fraction',
                 ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Output and Main
# -----------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    """Create output directories if they don't exist."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTARY_DIR.mkdir(parents=True, exist_ok=True)


def save_phase4_summary(
    cohort_results: List[CohortMatchResults],
    disparity_test: Dict[str, Any],
    variance_components: Optional[VarianceComponents]
) -> None:
    """Save summary of Phase 4 results."""
    summary_path = DATA_DIR / "phase4_results_summary.txt"

    all_pairs = [p for r in cohort_results for p in r.pairs]

    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 4: CROSS-BORDER ANALYSIS - RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("MODEL 4: CROSS-BORDER MATCHING\n")
        f.write("-" * 40 + "\n")
        f.write(f"Cohorts analyzed: {len(cohort_results)}\n")
        f.write(f"Total matched pairs: {len(all_pairs)}\n")
        f.write(f"\nDisparity Test (H0: E[Δ log fine] = 0):\n")
        f.write(f"  Mean Δ log fine: {disparity_test.get('mean_delta_log', 0):.4f}\n")
        f.write(f"  Std Δ log fine: {disparity_test.get('std_delta_log', 0):.4f}\n")
        f.write(f"  t-statistic: {disparity_test.get('t_stat', 0):.3f}\n")
        f.write(f"  p-value (one-sided): {disparity_test.get('p_value_onesided', 1):.4f}\n")
        f.write(f"  Cohen's d: {disparity_test.get('cohens_d', 0):.3f}\n")
        f.write(f"  Significant (p<0.05): {disparity_test.get('significant_005', False)}\n")

        if disparity_test.get('significant_005', False):
            f.write("\n  INTERPRETATION: H3 SUPPORTED - Significant cross-border disparity detected\n")
        else:
            f.write("\n  INTERPRETATION: H3 NOT SUPPORTED - No significant disparity\n")

        f.write("\n")

        f.write("MODEL 5: VARIANCE DECOMPOSITION\n")
        f.write("-" * 40 + "\n")
        if variance_components:
            f.write(f"Observations: {variance_components.n_obs}\n")
            f.write(f"Countries: {variance_components.n_countries}\n")
            f.write(f"Authorities: {variance_components.n_authorities}\n")
            f.write(f"\nVariance Components:\n")
            f.write(f"  σ²_country:   {variance_components.var_country:.4f} ({variance_components.pct_country:.1f}%)\n")
            f.write(f"  σ²_authority: {variance_components.var_authority:.4f} ({variance_components.pct_authority:.1f}%)\n")
            f.write(f"  σ²_residual:  {variance_components.var_residual:.4f} ({variance_components.pct_residual:.1f}%)\n")
            f.write(f"\nIntraclass Correlations:\n")
            f.write(f"  ICC (country): {variance_components.icc_country:.4f}\n")
            f.write(f"  ICC (country + authority): {variance_components.icc_authority:.4f}\n")

            if variance_components.pct_authority > 20:
                f.write("\n  INTERPRETATION: H4 SUPPORTED - Substantial authority-level heterogeneity\n")
                f.write(f"  \"Which DPA you get\" accounts for {variance_components.pct_authority:.1f}% of variance\n")
            else:
                f.write("\n  INTERPRETATION: Authority variance is modest\n")

            f.write(f"\nModel Note: {variance_components.model_notes}\n")
        else:
            f.write("Variance decomposition failed.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Generated by 8_cross_border_analysis.py (Phase 4)\n")

    logger.info(f"Saved Phase 4 summary to {summary_path}")


def main() -> None:
    """Execute Phase 4: Cross-Border Analysis."""
    logger.info("=" * 70)
    logger.info("PHASE 4: CROSS-BORDER ANALYSIS")
    logger.info("=" * 70)

    np.random.seed(RANDOM_SEED)
    ensure_output_dirs()

    # Load and prepare data
    logger.info("\n[1/7] Loading analysis sample...")
    df = load_analysis_sample()
    df = prepare_matching_data(df)

    # Cross-border matching
    logger.info("\n[2/7] Running cross-border matching (Model 4)...")
    cohort_results = run_cross_border_matching(df)

    # Disparity test
    all_pairs = [p for r in cohort_results for p in r.pairs]
    disparity_test = test_cross_border_disparity(all_pairs)
    logger.info(f"  Disparity test: t={disparity_test.get('t_stat', 0):.3f}, "
                f"p={disparity_test.get('p_value_onesided', 1):.4f}")

    # Table 6
    table6 = format_table6_cross_border_gaps(cohort_results, disparity_test)
    table6_path = TABLES_DIR / "table6_cross_border_gaps.csv"
    table6.to_csv(table6_path, index=False)
    logger.info(f"  Saved Table 6 to {table6_path}")
    print("\n" + "=" * 50)
    print("TABLE 6: Cross-Border Matched Pairs (Top Cohorts)")
    print("=" * 50)
    print(table6.head(15).to_string(index=False))

    # Variance decomposition
    logger.info("\n[3/7] Running variance decomposition (Model 5)...")
    variance_components = run_variance_decomposition(df)

    if variance_components:
        table7 = format_table7_variance_decomposition(variance_components)
        table7_path = TABLES_DIR / "table7_variance_decomposition.csv"
        table7.to_csv(table7_path, index=False)
        logger.info(f"  Saved Table 7 to {table7_path}")
        print("\n" + "=" * 50)
        print("TABLE 7: Variance Decomposition")
        print("=" * 50)
        print(table7.to_string(index=False))

    # Figure 4
    logger.info("\n[4/7] Creating Figure 4: Matched-pair distributions...")
    fig4 = create_figure4_matched_pairs(cohort_results, disparity_test)
    if fig4:
        fig4_path = FIGURES_DIR / "figure4_cross_border_gaps.png"
        fig4.savefig(fig4_path, dpi=300, bbox_inches="tight")
        fig4.savefig(FIGURES_DIR / "figure4_cross_border_gaps.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig4)
        logger.info(f"  Saved Figure 4 to {fig4_path}")

    # Figure 5
    logger.info("\n[5/7] Creating Figure 5: Variance partition...")
    if variance_components:
        fig5 = create_figure5_variance_partition(variance_components)
        if fig5:
            fig5_path = FIGURES_DIR / "figure5_variance_partition.png"
            fig5.savefig(fig5_path, dpi=300, bbox_inches="tight")
            fig5.savefig(FIGURES_DIR / "figure5_variance_partition.pdf", format="pdf", bbox_inches="tight")
            plt.close(fig5)
            logger.info(f"  Saved Figure 5 to {fig5_path}")

    # Mega-fine sensitivity analysis (NEW)
    logger.info("\n[6/7] Running mega-fine sensitivity analysis...")
    mega_sensitivity_results = run_cross_border_mega_fine_sensitivity(
        cohort_results, disparity_test, df
    )

    if mega_sensitivity_results:
        mega_table = format_table_mega_fine_sensitivity(mega_sensitivity_results)
        mega_path = SUPPLEMENTARY_DIR / "tableS7_cross_border_mega_fine_sensitivity.csv"
        mega_table.to_csv(mega_path, index=False)
        logger.info(f"  Saved Table S7 to {mega_path}")
        print("\n" + "=" * 50)
        print("TABLE S7: Cross-Border Mega-Fine Sensitivity")
        print("=" * 50)
        print(mega_table.to_string(index=False))

    # Save matched pairs data
    logger.info("\n[7/7] Saving matched pairs dataset...")
    pairs_records = [
        {
            "case_i_id": p.case_i_id,
            "case_j_id": p.case_j_id,
            "country_i": p.country_i,
            "country_j": p.country_j,
            "article_cohort": p.article_cohort,
            "log_fine_i": p.log_fine_i,
            "log_fine_j": p.log_fine_j,
            "fine_eur_i": p.fine_eur_i,
            "fine_eur_j": p.fine_eur_j,
            "delta_log_fine": p.delta_log_fine,
            "delta_fine_eur": p.delta_fine_eur,
            "distance": p.distance,
            "defendant_class_match": p.defendant_class_match,
            "enterprise_size_match": p.enterprise_size_match,
            "year_diff": p.year_diff,
        }
        for p in all_pairs
    ]
    pairs_df = pd.DataFrame.from_records(pairs_records)
    pairs_path = DATA_DIR / "matched_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False)
    logger.info(f"  Saved {len(pairs_df)} matched pairs to {pairs_path}")

    # Save summary
    save_phase4_summary(cohort_results, disparity_test, variance_components)

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4 COMPLETE")
    logger.info("=" * 70)

    print("\n" + "=" * 50)
    print("PHASE 4 SUMMARY")
    print("=" * 50)
    print(f"\nCross-Border Matching (Model 4):")
    print(f"  Cohorts analyzed: {len(cohort_results)}")
    print(f"  Total matched pairs: {len(all_pairs)}")
    print(f"  Mean Δ log fine: {disparity_test.get('mean_delta_log', 0):.3f}")
    print(f"  t-statistic: {disparity_test.get('t_stat', 0):.3f}")
    print(f"  p-value (one-sided): {disparity_test.get('p_value_onesided', 1):.4f}")

    sig = "***" if disparity_test.get('p_value_onesided', 1) < 0.001 else \
          "**" if disparity_test.get('p_value_onesided', 1) < 0.01 else \
          "*" if disparity_test.get('p_value_onesided', 1) < 0.05 else ""
    print(f"  Significant disparity: {'YES' if sig else 'NO'} {sig}")

    if variance_components:
        print(f"\nVariance Decomposition (Model 5):")
        print(f"  Country variance: {variance_components.pct_country:.1f}%")
        print(f"  Authority variance: {variance_components.pct_authority:.1f}%")
        print(f"  Case-level variance: {variance_components.pct_residual:.1f}%")
        print(f"  ICC (authority level): {variance_components.icc_authority:.3f}")

    if mega_sensitivity_results:
        print(f"\nMega-Fine Sensitivity (Outlier Robustness):")
        disparity_rows = [r for r in mega_sensitivity_results if r.metric_name == "Implied fine ratio (exp)"]
        for r in disparity_rows:
            print(f"  {r.sample_description}: {r.full_sample_value:.1f}x → {r.restricted_value:.1f}x ({r.pct_change:+.1f}%)")

    print("\n" + "=" * 50)
    print(f"Outputs saved to:")
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Supplementary: {SUPPLEMENTARY_DIR}")
    print(f"  Data: {DATA_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
