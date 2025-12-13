"""Phase 4 – Cross-Border Analysis: Matching and Variance Decomposition.

This script implements Phase 4 (Cross-Border Analysis) of the methodology proposal:
"Reasoning and Consistency in GDPR Enforcement: A Cross-Border Analysis"

Analyses Implemented:
1. Nearest-neighbor matching within article cohorts (RQ3)
2. Gap computation for matched pairs (cross-border fine disparities)
3. Model 5: Variance decomposition via nested random effects (RQ4)

Outputs:
- Table 6: Cross-border matched pairs - mean gaps by article cohort
- Table 7: Variance decomposition - country vs. authority vs. case
- Figure 4: Matched-pair fine gap distributions
- Figure 5: Variance partition visualization

Input:  outputs/paper/data/analysis_sample.csv
        outputs/paper/data/cohort_membership.csv
Output: outputs/paper/tables/
        outputs/paper/figures/
        outputs/paper/data/
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist, mahalanobis

# Suppress convergence warnings for display
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Optional imports with fallbacks
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Install with: pip install statsmodels")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for CLI
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not installed. Figures will be skipped.")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Input paths
ANALYSIS_SAMPLE_PATH = Path("outputs/paper/data/analysis_sample.csv")
COHORT_MEMBERSHIP_PATH = Path("outputs/paper/data/cohort_membership.csv")

# Output directories
TABLES_DIR = Path("outputs/paper/tables")
FIGURES_DIR = Path("outputs/paper/figures")
DATA_DIR = Path("outputs/paper/data")

# Random seed for reproducibility
RANDOM_SEED = 42

# Minimum cases per cohort for meaningful matching
MIN_CASES_PER_COHORT = 2
MIN_COUNTRIES_PER_COHORT = 2

# Matching variables (for Mahalanobis distance)
MATCHING_VARS = [
    "defendant_class_code",
    "enterprise_size_code",
    "sector_code",
    "decision_year",
]

# Matching caliper (maximum Mahalanobis distance for valid match)
DEFAULT_CALIPER = None  # No caliper by default; use sensitivity analysis

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Classes for Results
# -----------------------------------------------------------------------------

@dataclass
class MatchedPair:
    """Container for a single matched pair."""
    case_i_id: str
    case_j_id: str
    country_i: str
    country_j: str
    article_set_key: str
    log_fine_i: float
    log_fine_j: float
    fine_eur_i: float
    fine_eur_j: float
    delta_log_fine: float  # |log_fine_i - log_fine_j|
    delta_fine_eur: float  # |fine_eur_i - fine_eur_j|
    mahalanobis_distance: float


@dataclass
class CohortMatchingResults:
    """Container for matching results within a cohort."""
    article_set_key: str
    n_cases: int
    n_countries: int
    n_matched_pairs: int
    countries: List[str]
    mean_delta_log_fine: float
    std_delta_log_fine: float
    median_delta_log_fine: float
    mean_delta_fine_eur: float
    median_delta_fine_eur: float
    matched_pairs: List[MatchedPair]


@dataclass
class VarianceDecomposition:
    """Container for variance decomposition results."""
    n_obs: int
    n_countries: int
    n_authorities: int
    var_country: float
    var_authority: float  # Within-country
    var_residual: float  # Within-authority (case-level)
    var_total: float
    icc_country: float  # Country-level ICC
    icc_authority: float  # Authority-level ICC (within country)
    icc_combined: float  # Country + Authority ICC
    pct_country: float
    pct_authority: float
    pct_residual: float


# -----------------------------------------------------------------------------
# Data Loading and Preparation
# -----------------------------------------------------------------------------

def load_analysis_sample() -> pd.DataFrame:
    """Load the analytical sample from Phase 1."""
    if not ANALYSIS_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Analysis sample not found at {ANALYSIS_SAMPLE_PATH}. "
            "Run Phase 1 (6_paper_data_preparation.py) first."
        )
    df = pd.read_csv(ANALYSIS_SAMPLE_PATH, low_memory=False)
    logger.info(f"Loaded analysis sample: {len(df)} rows")
    return df


def load_cohort_membership() -> pd.DataFrame:
    """Load cohort membership from Phase 1."""
    if not COHORT_MEMBERSHIP_PATH.exists():
        raise FileNotFoundError(
            f"Cohort membership not found at {COHORT_MEMBERSHIP_PATH}. "
            "Run Phase 1 (6_paper_data_preparation.py) first."
        )
    df = pd.read_csv(COHORT_MEMBERSHIP_PATH)
    logger.info(f"Loaded cohort membership: {len(df)} cohorts")
    return df


def prepare_matching_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for matching analysis.

    - Encode categorical variables as numeric
    - Filter to cross-border eligible cases
    - Handle missing values
    """
    data = df.copy()

    # Ensure required columns exist
    if "log_fine_2025" not in data.columns:
        fine_col = "fine_amount_eur_real_2025" if "fine_amount_eur_real_2025" in data.columns else "fine_amount_eur"
        data["log_fine_2025"] = np.log1p(data[fine_col].clip(lower=0).fillna(0))

    # Create case identifier
    if "id" not in data.columns:
        data["case_id"] = data.index.astype(str)
    else:
        data["case_id"] = data["id"].astype(str)

    # Encode defendant class
    class_map = {
        "PRIVATE": 1, "PUBLIC": 2, "NGO_OR_NONPROFIT": 3,
        "RELIGIOUS": 4, "INDIVIDUAL": 5, "POLITICAL_PARTY": 6, "OTHER": 7
    }
    data["defendant_class_code"] = data["a8_defendant_class"].map(class_map).fillna(0)

    # Encode enterprise size
    size_map = {"SME": 1, "LARGE": 2, "VERY_LARGE": 3, "UNKNOWN": 0, "NOT_APPLICABLE": 0}
    data["enterprise_size_code"] = data["a9_enterprise_size"].map(size_map).fillna(0)

    # Encode sector (using hash for categorical)
    if "a12_sector" in data.columns:
        unique_sectors = data["a12_sector"].dropna().unique()
        sector_map = {s: i+1 for i, s in enumerate(sorted(unique_sectors))}
        data["sector_code"] = data["a12_sector"].map(sector_map).fillna(0)
    else:
        data["sector_code"] = 0

    # Decision year (already numeric, just ensure no NaN)
    data["decision_year"] = pd.to_numeric(data["decision_year"], errors="coerce").fillna(2020)

    # Country code standardization
    if "a1_country_code" in data.columns:
        data["country_code"] = data["a1_country_code"]

    # Authority identifier
    if "authority_name_norm" not in data.columns:
        data["authority_name_norm"] = data["a2_authority_name"]

    # Filter to cross-border eligible
    if "cross_border_eligible" in data.columns:
        eligible_count = data["cross_border_eligible"].sum()
        logger.info(f"Cross-border eligible cases: {eligible_count}")

    # Fine amount in EUR (for gap calculation)
    fine_col = "fine_amount_eur_real_2025" if "fine_amount_eur_real_2025" in data.columns else "fine_amount_eur"
    data["fine_amount_eur_analysis"] = data[fine_col].fillna(0)

    logger.info(f"Prepared matching data: {len(data)} cases")
    return data


# -----------------------------------------------------------------------------
# Nearest-Neighbor Matching
# -----------------------------------------------------------------------------

def compute_mahalanobis_matrix(
    X: np.ndarray,
    regularization: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Mahalanobis distance matrix.

    Args:
        X: n x p matrix of matching variables
        regularization: Ridge parameter for covariance stability

    Returns:
        Tuple of (covariance matrix, inverse covariance matrix)
    """
    # Compute covariance with regularization
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[cov]])

    # Add ridge regularization for invertibility
    cov_reg = cov + regularization * np.eye(cov.shape[0])

    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        cov_inv = np.linalg.pinv(cov_reg)

    return cov, cov_inv


def find_nearest_neighbor(
    target_row: pd.Series,
    candidate_df: pd.DataFrame,
    matching_vars: List[str],
    cov_inv: np.ndarray,
    caliper: Optional[float] = None
) -> Optional[Tuple[str, float]]:
    """
    Find nearest neighbor for target case among candidates.

    Args:
        target_row: Single case to match
        candidate_df: Potential matches (different country)
        matching_vars: Variables for Mahalanobis distance
        cov_inv: Inverse covariance matrix
        caliper: Maximum distance for valid match

    Returns:
        Tuple of (matched_case_id, mahalanobis_distance) or None
    """
    if len(candidate_df) == 0:
        return None

    # Extract target features
    target_x = target_row[matching_vars].values.astype(float)

    # Extract candidate features
    candidate_x = candidate_df[matching_vars].values.astype(float)

    # Compute Mahalanobis distances
    distances = []
    for i in range(len(candidate_x)):
        diff = target_x - candidate_x[i]
        dist = np.sqrt(diff @ cov_inv @ diff)
        distances.append(dist)

    distances = np.array(distances)

    # Find minimum
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]

    # Apply caliper if specified
    if caliper is not None and min_dist > caliper:
        return None

    matched_id = candidate_df.iloc[min_idx]["case_id"]
    return (matched_id, min_dist)


def match_within_cohort(
    cohort_df: pd.DataFrame,
    matching_vars: List[str],
    caliper: Optional[float] = None
) -> List[MatchedPair]:
    """
    Perform nearest-neighbor matching within a single article cohort.

    For each case in country c, find nearest neighbor in country c' != c.
    Uses Mahalanobis distance on matching variables.

    Args:
        cohort_df: Cases in this article cohort
        matching_vars: Variables for distance computation
        caliper: Maximum distance for valid match

    Returns:
        List of matched pairs
    """
    if len(cohort_df) < 2:
        return []

    countries = cohort_df["country_code"].unique()
    if len(countries) < 2:
        return []

    # Filter to cases with valid matching variables
    valid_vars = [v for v in matching_vars if v in cohort_df.columns]
    if not valid_vars:
        return []

    cohort_df = cohort_df.dropna(subset=valid_vars + ["log_fine_2025"])
    if len(cohort_df) < 2:
        return []

    # Compute covariance matrix
    X = cohort_df[valid_vars].values.astype(float)
    _, cov_inv = compute_mahalanobis_matrix(X)

    matched_pairs: List[MatchedPair] = []
    matched_j_ids = set()  # Track used matches to avoid duplicates

    for _, row_i in cohort_df.iterrows():
        country_i = row_i["country_code"]

        # Candidates: different country, not already matched
        candidates = cohort_df[
            (cohort_df["country_code"] != country_i) &
            (~cohort_df["case_id"].isin(matched_j_ids))
        ]

        if len(candidates) == 0:
            continue

        # Find nearest neighbor
        match_result = find_nearest_neighbor(row_i, candidates, valid_vars, cov_inv, caliper)

        if match_result is None:
            continue

        matched_id, distance = match_result
        row_j = cohort_df[cohort_df["case_id"] == matched_id].iloc[0]

        # Compute fine gaps
        delta_log = abs(row_i["log_fine_2025"] - row_j["log_fine_2025"])
        delta_eur = abs(row_i["fine_amount_eur_analysis"] - row_j["fine_amount_eur_analysis"])

        pair = MatchedPair(
            case_i_id=str(row_i["case_id"]),
            case_j_id=str(matched_id),
            country_i=country_i,
            country_j=row_j["country_code"],
            article_set_key=row_i["article_set_key"],
            log_fine_i=row_i["log_fine_2025"],
            log_fine_j=row_j["log_fine_2025"],
            fine_eur_i=row_i["fine_amount_eur_analysis"],
            fine_eur_j=row_j["fine_amount_eur_analysis"],
            delta_log_fine=delta_log,
            delta_fine_eur=delta_eur,
            mahalanobis_distance=distance,
        )

        matched_pairs.append(pair)
        matched_j_ids.add(matched_id)

    return matched_pairs


def run_cross_border_matching(
    df: pd.DataFrame,
    caliper: Optional[float] = None
) -> List[CohortMatchingResults]:
    """
    Run nearest-neighbor matching across all cross-border eligible cohorts.

    Args:
        df: Prepared analysis sample
        caliper: Maximum Mahalanobis distance for valid match

    Returns:
        List of cohort-level matching results
    """
    results: List[CohortMatchingResults] = []

    # Filter to cross-border eligible
    if "cross_border_eligible" in df.columns:
        eligible_df = df[df["cross_border_eligible"] == True].copy()
    else:
        # Compute eligibility
        cohort_countries = df.groupby("article_set_key")["country_code"].nunique()
        eligible_cohorts = cohort_countries[cohort_countries >= 2].index
        eligible_df = df[df["article_set_key"].isin(eligible_cohorts)].copy()

    logger.info(f"Cross-border eligible cases: {len(eligible_df)}")

    # Get matching variables
    valid_matching_vars = [v for v in MATCHING_VARS if v in eligible_df.columns]
    logger.info(f"Matching variables: {valid_matching_vars}")

    # Process each cohort
    cohorts = eligible_df.groupby("article_set_key")
    n_cohorts = len(cohorts)

    for article_key, cohort_df in cohorts:
        countries = sorted(cohort_df["country_code"].dropna().unique())
        n_countries = len(countries)

        if n_countries < MIN_COUNTRIES_PER_COHORT:
            continue

        if len(cohort_df) < MIN_CASES_PER_COHORT:
            continue

        # Run matching
        matched_pairs = match_within_cohort(cohort_df, valid_matching_vars, caliper)

        if not matched_pairs:
            continue

        # Compute summary statistics
        deltas_log = [p.delta_log_fine for p in matched_pairs]
        deltas_eur = [p.delta_fine_eur for p in matched_pairs]

        results.append(CohortMatchingResults(
            article_set_key=str(article_key),
            n_cases=len(cohort_df),
            n_countries=n_countries,
            n_matched_pairs=len(matched_pairs),
            countries=countries,
            mean_delta_log_fine=np.mean(deltas_log),
            std_delta_log_fine=np.std(deltas_log, ddof=1) if len(deltas_log) > 1 else 0,
            median_delta_log_fine=np.median(deltas_log),
            mean_delta_fine_eur=np.mean(deltas_eur),
            median_delta_fine_eur=np.median(deltas_eur),
            matched_pairs=matched_pairs,
        ))

    logger.info(f"Matching complete: {len(results)} cohorts, "
                f"{sum(r.n_matched_pairs for r in results)} total pairs")
    return results


def aggregate_matched_pairs(results: List[CohortMatchingResults]) -> pd.DataFrame:
    """Aggregate all matched pairs into a single DataFrame."""
    records = []
    for cohort in results:
        for pair in cohort.matched_pairs:
            records.append({
                "article_set_key": pair.article_set_key,
                "case_i_id": pair.case_i_id,
                "case_j_id": pair.case_j_id,
                "country_i": pair.country_i,
                "country_j": pair.country_j,
                "log_fine_i": pair.log_fine_i,
                "log_fine_j": pair.log_fine_j,
                "fine_eur_i": pair.fine_eur_i,
                "fine_eur_j": pair.fine_eur_j,
                "delta_log_fine": pair.delta_log_fine,
                "delta_fine_eur": pair.delta_fine_eur,
                "mahalanobis_distance": pair.mahalanobis_distance,
            })
    return pd.DataFrame.from_records(records)


# -----------------------------------------------------------------------------
# Statistical Tests for Cross-Border Disparities
# -----------------------------------------------------------------------------

def test_cross_border_disparities(
    matched_pairs_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Test H3: Whether cross-border disparities are significant.

    Tests:
    1. One-sample t-test: E[delta_log_fine] > 0
    2. Wilcoxon signed-rank test (non-parametric)
    3. Meta-analytic aggregation across cohorts
    """
    results = {}

    deltas = matched_pairs_df["delta_log_fine"].dropna()
    n_pairs = len(deltas)

    if n_pairs < 3:
        logger.warning("Insufficient pairs for statistical tests")
        return results

    # Descriptive statistics
    results["n_pairs"] = n_pairs
    results["mean_delta"] = deltas.mean()
    results["std_delta"] = deltas.std()
    results["median_delta"] = deltas.median()

    # One-sample t-test: H0: E[delta] = 0 vs H1: E[delta] > 0
    t_stat, p_value_two = stats.ttest_1samp(deltas, 0)
    results["t_stat"] = t_stat
    results["t_pvalue_onesided"] = p_value_two / 2 if t_stat > 0 else 1 - p_value_two / 2

    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, w_pvalue = stats.wilcoxon(deltas, alternative="greater")
        results["wilcoxon_stat"] = w_stat
        results["wilcoxon_pvalue"] = w_pvalue
    except Exception:
        results["wilcoxon_stat"] = np.nan
        results["wilcoxon_pvalue"] = np.nan

    # Effect size: Cohen's d
    results["cohens_d"] = deltas.mean() / deltas.std() if deltas.std() > 0 else 0

    # 95% CI for mean delta
    se = deltas.std() / np.sqrt(n_pairs)
    t_crit = stats.t.ppf(0.975, n_pairs - 1)
    results["ci_lower"] = deltas.mean() - t_crit * se
    results["ci_upper"] = deltas.mean() + t_crit * se

    logger.info(f"Cross-border disparity test: mean_delta={results['mean_delta']:.4f}, "
                f"t={results['t_stat']:.3f}, p={results['t_pvalue_onesided']:.4f}")

    return results


# -----------------------------------------------------------------------------
# Model 5: Variance Decomposition (Nested Random Effects)
# -----------------------------------------------------------------------------

def run_variance_decomposition(df: pd.DataFrame) -> Optional[VarianceDecomposition]:
    """
    Run Model 5: Three-level variance decomposition.

    Specification:
    log_fine_2025_ijk = β₀ + γ·X + v_c[k] + u_a[j,k] + ε_ijk

    Where:
    - k indexes countries
    - j indexes authorities within countries
    - i indexes cases within authorities
    - v_c = country random intercept
    - u_a = authority random intercept (nested within country)

    Returns variance partition: country vs. authority vs. case
    """
    if not HAS_STATSMODELS:
        logger.error("statsmodels required for variance decomposition")
        return None

    # Prepare data
    model_df = df[["log_fine_2025", "country_code", "authority_name_norm"]].dropna()

    # Need multiple observations per authority for meaningful decomposition
    auth_counts = model_df.groupby("authority_name_norm").size()
    valid_auths = auth_counts[auth_counts >= 2].index
    model_df = model_df[model_df["authority_name_norm"].isin(valid_auths)]

    n_obs = len(model_df)
    n_countries = model_df["country_code"].nunique()
    n_authorities = model_df["authority_name_norm"].nunique()

    logger.info(f"Variance decomposition sample: {n_obs} obs, "
                f"{n_countries} countries, {n_authorities} authorities")

    if n_obs < 50 or n_authorities < 5:
        logger.warning("Insufficient data for variance decomposition")
        return None

    try:
        # Model 1: Country-level random effects only
        model_country = smf.mixedlm(
            "log_fine_2025 ~ 1",
            data=model_df,
            groups=model_df["country_code"],
            re_formula="~1"
        )
        fit_country = model_country.fit(method="powell", maxiter=500)

        var_country = float(fit_country.cov_re.iloc[0, 0]) if hasattr(fit_country.cov_re, 'iloc') else float(fit_country.cov_re)
        var_resid_country = fit_country.scale

        # Model 2: Authority-level random effects only (ignoring country nesting for comparison)
        model_authority = smf.mixedlm(
            "log_fine_2025 ~ 1",
            data=model_df,
            groups=model_df["authority_name_norm"],
            re_formula="~1"
        )
        fit_authority = model_authority.fit(method="powell", maxiter=500)

        var_authority = float(fit_authority.cov_re.iloc[0, 0]) if hasattr(fit_authority.cov_re, 'iloc') else float(fit_authority.cov_re)
        var_resid_authority = fit_authority.scale

        # For true nested decomposition, we need to account for the hierarchy
        # Approach: Use country as a fixed effect, authority as random
        # This gives us within-country authority variance

        # Create country dummies for fixed effects
        model_df["country_factor"] = pd.Categorical(model_df["country_code"])

        try:
            # Nested model: authority random effect with country fixed effect
            formula_nested = "log_fine_2025 ~ C(country_code)"
            model_nested = smf.mixedlm(
                formula_nested,
                data=model_df,
                groups=model_df["authority_name_norm"],
                re_formula="~1"
            )
            fit_nested = model_nested.fit(method="powell", maxiter=500)

            var_auth_within = float(fit_nested.cov_re.iloc[0, 0]) if hasattr(fit_nested.cov_re, 'iloc') else float(fit_nested.cov_re)
            var_resid_nested = fit_nested.scale
        except Exception as e:
            logger.warning(f"Nested model failed, using approximation: {e}")
            var_auth_within = var_authority
            var_resid_nested = var_resid_authority

        # Compute total variance
        total_var = model_df["log_fine_2025"].var()

        # Approximate decomposition
        # Country variance from country-only model
        # Authority variance (within-country) from nested model
        # Residual variance from nested model
        var_total = var_country + var_auth_within + var_resid_nested

        # ICC calculations
        icc_country = var_country / var_total if var_total > 0 else 0
        icc_authority = var_auth_within / var_total if var_total > 0 else 0
        icc_combined = (var_country + var_auth_within) / var_total if var_total > 0 else 0

        # Percentages
        pct_country = var_country / var_total * 100 if var_total > 0 else 0
        pct_authority = var_auth_within / var_total * 100 if var_total > 0 else 0
        pct_residual = var_resid_nested / var_total * 100 if var_total > 0 else 0

        return VarianceDecomposition(
            n_obs=n_obs,
            n_countries=n_countries,
            n_authorities=n_authorities,
            var_country=var_country,
            var_authority=var_auth_within,
            var_residual=var_resid_nested,
            var_total=var_total,
            icc_country=icc_country,
            icc_authority=icc_authority,
            icc_combined=icc_combined,
            pct_country=pct_country,
            pct_authority=pct_authority,
            pct_residual=pct_residual,
        )

    except Exception as e:
        logger.error(f"Variance decomposition failed: {e}")
        return None


# -----------------------------------------------------------------------------
# Output Generation: Tables
# -----------------------------------------------------------------------------

def format_table6_matching_results(
    cohort_results: List[CohortMatchingResults]
) -> pd.DataFrame:
    """Format Table 6: Cross-border matched pairs by article cohort."""
    rows = []

    # Sort by number of matched pairs (descending)
    sorted_results = sorted(cohort_results, key=lambda x: x.n_matched_pairs, reverse=True)

    for r in sorted_results:
        rows.append({
            "Article Cohort": r.article_set_key,
            "N Cases": r.n_cases,
            "N Countries": r.n_countries,
            "Countries": ";".join(r.countries[:5]) + ("..." if len(r.countries) > 5 else ""),
            "N Pairs": r.n_matched_pairs,
            "Mean Δ(log fine)": f"{r.mean_delta_log_fine:.3f}",
            "SD Δ(log fine)": f"{r.std_delta_log_fine:.3f}",
            "Median Δ(€)": f"€{r.median_delta_fine_eur:,.0f}",
        })

    # Add summary row
    total_pairs = sum(r.n_matched_pairs for r in cohort_results)
    all_deltas = [p.delta_log_fine for r in cohort_results for p in r.matched_pairs]
    if all_deltas:
        rows.append({
            "Article Cohort": "TOTAL",
            "N Cases": sum(r.n_cases for r in cohort_results),
            "N Countries": "—",
            "Countries": "—",
            "N Pairs": total_pairs,
            "Mean Δ(log fine)": f"{np.mean(all_deltas):.3f}",
            "SD Δ(log fine)": f"{np.std(all_deltas):.3f}",
            "Median Δ(€)": "—",
        })

    return pd.DataFrame(rows)


def format_table7_variance_decomposition(
    decomp: VarianceDecomposition
) -> pd.DataFrame:
    """Format Table 7: Variance decomposition results."""
    rows = [
        {
            "Component": "Country-Level",
            "Variance (σ²)": f"{decomp.var_country:.4f}",
            "% of Total": f"{decomp.pct_country:.1f}%",
            "ICC": f"{decomp.icc_country:.4f}",
            "Interpretation": "Between-country heterogeneity"
        },
        {
            "Component": "Authority-Level (within country)",
            "Variance (σ²)": f"{decomp.var_authority:.4f}",
            "% of Total": f"{decomp.pct_authority:.1f}%",
            "ICC": f"{decomp.icc_authority:.4f}",
            "Interpretation": "Within-country, between-authority variation"
        },
        {
            "Component": "Case-Level (residual)",
            "Variance (σ²)": f"{decomp.var_residual:.4f}",
            "% of Total": f"{decomp.pct_residual:.1f}%",
            "ICC": "—",
            "Interpretation": "Within-authority, case-to-case variation"
        },
        {
            "Component": "Total",
            "Variance (σ²)": f"{decomp.var_total:.4f}",
            "% of Total": "100%",
            "ICC": f"{decomp.icc_combined:.4f}",
            "Interpretation": "Combined country + authority ICC"
        },
    ]

    # Add sample info
    rows.append({
        "Component": "---",
        "Variance (σ²)": "---",
        "% of Total": "---",
        "ICC": "---",
        "Interpretation": "---"
    })
    rows.append({
        "Component": "N (observations)",
        "Variance (σ²)": str(decomp.n_obs),
        "% of Total": "",
        "ICC": "",
        "Interpretation": ""
    })
    rows.append({
        "Component": "N (countries)",
        "Variance (σ²)": str(decomp.n_countries),
        "% of Total": "",
        "ICC": "",
        "Interpretation": ""
    })
    rows.append({
        "Component": "N (authorities)",
        "Variance (σ²)": str(decomp.n_authorities),
        "% of Total": "",
        "ICC": "",
        "Interpretation": ""
    })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Output Generation: Figures
# -----------------------------------------------------------------------------

def create_figure4_matched_pairs_distribution(
    matched_pairs_df: pd.DataFrame,
    test_results: Dict[str, Any]
) -> Optional[plt.Figure]:
    """Create Figure 4: Matched-pair fine gap distributions."""
    if not HAS_PLOTTING:
        logger.warning("Plotting libraries not available")
        return None

    if len(matched_pairs_df) < 3:
        logger.warning("Insufficient pairs for distribution plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Log fine gaps
    ax1 = axes[0]
    deltas_log = matched_pairs_df["delta_log_fine"].dropna()

    sns.histplot(deltas_log, bins=30, kde=True, ax=ax1, color="steelblue", alpha=0.7)
    ax1.axvline(deltas_log.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {deltas_log.mean():.3f}')
    ax1.axvline(deltas_log.median(), color='orange', linestyle=':', linewidth=2,
                label=f'Median = {deltas_log.median():.3f}')

    ax1.set_xlabel("Δ(log fine) = |log(fine_i) - log(fine_j)|", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("A. Distribution of Log Fine Gaps\n(Cross-Border Matched Pairs)", fontsize=12)
    ax1.legend(loc="upper right")

    # Add test results annotation
    if test_results:
        test_text = (f"n = {test_results.get('n_pairs', 0)} pairs\n"
                    f"t = {test_results.get('t_stat', 0):.3f}\n"
                    f"p = {test_results.get('t_pvalue_onesided', 1):.4f}\n"
                    f"Cohen's d = {test_results.get('cohens_d', 0):.3f}")
        ax1.text(0.95, 0.95, test_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel B: EUR fine gaps (log scale)
    ax2 = axes[1]
    deltas_eur = matched_pairs_df["delta_fine_eur"].dropna()
    deltas_eur_log = np.log10(deltas_eur + 1)  # Log scale for visualization

    sns.histplot(deltas_eur_log, bins=30, kde=True, ax=ax2, color="forestgreen", alpha=0.7)
    ax2.axvline(np.log10(deltas_eur.mean() + 1), color='red', linestyle='--', linewidth=2,
                label=f'Mean = €{deltas_eur.mean():,.0f}')
    ax2.axvline(np.log10(deltas_eur.median() + 1), color='orange', linestyle=':', linewidth=2,
                label=f'Median = €{deltas_eur.median():,.0f}')

    ax2.set_xlabel("log₁₀(Δ EUR + 1)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("B. Distribution of EUR Fine Gaps\n(Log Scale)", fontsize=12)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    return fig


def create_figure5_variance_partition(
    decomp: VarianceDecomposition
) -> Optional[plt.Figure]:
    """Create Figure 5: Variance partition pie chart."""
    if not HAS_PLOTTING:
        logger.warning("Plotting libraries not available")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Pie chart
    ax1 = axes[0]
    sizes = [decomp.pct_country, decomp.pct_authority, decomp.pct_residual]
    labels = ['Country', 'Authority\n(within country)', 'Case\n(residual)']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.05, 0.05, 0)

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.6,
        textprops={'fontsize': 10}
    )
    ax1.set_title("A. Variance Partition\n(Fine Magnitude)", fontsize=12)

    # Panel B: Bar chart with ICC values
    ax2 = axes[1]
    components = ['Country\n(between)', 'Authority\n(within country)', 'Residual\n(case-level)']
    variances = [decomp.var_country, decomp.var_authority, decomp.var_residual]

    bars = ax2.bar(components, variances, color=colors, edgecolor='black', alpha=0.8)

    # Add ICC annotations
    ax2.text(0, decomp.var_country + 0.1, f'ICC={decomp.icc_country:.3f}',
             ha='center', fontsize=9)
    ax2.text(1, decomp.var_authority + 0.1, f'ICC={decomp.icc_authority:.3f}',
             ha='center', fontsize=9)

    ax2.set_ylabel("Variance (σ²)", fontsize=11)
    ax2.set_title("B. Variance Components\nwith ICC Values", fontsize=12)
    ax2.set_ylim(0, max(variances) * 1.3)

    # Add combined ICC annotation
    ax2.text(0.5, 0.95, f"Combined ICC (Country + Authority) = {decomp.icc_combined:.3f}",
             transform=ax2.transAxes, fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    """Create output directories if they don't exist."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directories ready")


def save_phase4_summary(
    matching_results: List[CohortMatchingResults],
    test_results: Dict[str, Any],
    variance_results: Optional[VarianceDecomposition]
) -> None:
    """Save comprehensive Phase 4 results summary."""
    summary_path = DATA_DIR / "phase4_results_summary.txt"

    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 4: CROSS-BORDER ANALYSIS - RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        # Matching Summary
        f.write("CROSS-BORDER MATCHING (RQ3)\n")
        f.write("-" * 40 + "\n")
        total_pairs = sum(r.n_matched_pairs for r in matching_results)
        all_deltas = [p.delta_log_fine for r in matching_results for p in r.matched_pairs]

        f.write(f"Cohorts analyzed: {len(matching_results)}\n")
        f.write(f"Total matched pairs: {total_pairs}\n")

        if all_deltas:
            f.write(f"\nFine Gap Statistics:\n")
            f.write(f"  Mean Δ(log fine): {np.mean(all_deltas):.4f}\n")
            f.write(f"  Std Δ(log fine): {np.std(all_deltas):.4f}\n")
            f.write(f"  Median Δ(log fine): {np.median(all_deltas):.4f}\n")

        f.write("\n")

        # Statistical Tests
        f.write("DISPARITY TESTS (H3)\n")
        f.write("-" * 40 + "\n")
        if test_results:
            f.write(f"N pairs: {test_results.get('n_pairs', 0)}\n")
            f.write(f"t-statistic: {test_results.get('t_stat', 0):.4f}\n")
            f.write(f"p-value (one-sided): {test_results.get('t_pvalue_onesided', 1):.4f}\n")
            f.write(f"Cohen's d: {test_results.get('cohens_d', 0):.4f}\n")
            f.write(f"95% CI: [{test_results.get('ci_lower', 0):.4f}, "
                    f"{test_results.get('ci_upper', 0):.4f}]\n")

            sig = "***" if test_results.get('t_pvalue_onesided', 1) < 0.001 else \
                  "**" if test_results.get('t_pvalue_onesided', 1) < 0.01 else \
                  "*" if test_results.get('t_pvalue_onesided', 1) < 0.05 else ""

            f.write(f"\nInterpretation: ")
            if test_results.get('t_pvalue_onesided', 1) < 0.05:
                f.write(f"Cross-border disparities are SIGNIFICANT{sig}\n")
                f.write("H3 SUPPORTED: Similar violations receive different penalties across jurisdictions\n")
            else:
                f.write("Cross-border disparities are NOT SIGNIFICANT\n")
                f.write("H3 NOT SUPPORTED at α=0.05\n")

        f.write("\n")

        # Variance Decomposition
        f.write("VARIANCE DECOMPOSITION (RQ4)\n")
        f.write("-" * 40 + "\n")
        if variance_results:
            f.write(f"Observations: {variance_results.n_obs}\n")
            f.write(f"Countries: {variance_results.n_countries}\n")
            f.write(f"Authorities: {variance_results.n_authorities}\n")
            f.write(f"\nVariance Components:\n")
            f.write(f"  σ²_country: {variance_results.var_country:.4f} "
                    f"({variance_results.pct_country:.1f}%)\n")
            f.write(f"  σ²_authority: {variance_results.var_authority:.4f} "
                    f"({variance_results.pct_authority:.1f}%)\n")
            f.write(f"  σ²_residual: {variance_results.var_residual:.4f} "
                    f"({variance_results.pct_residual:.1f}%)\n")
            f.write(f"\nIntraclass Correlations:\n")
            f.write(f"  ICC (country): {variance_results.icc_country:.4f}\n")
            f.write(f"  ICC (authority, within country): {variance_results.icc_authority:.4f}\n")
            f.write(f"  ICC (combined): {variance_results.icc_combined:.4f}\n")

            f.write(f"\nInterpretation:\n")
            if variance_results.icc_authority > 0.1:
                f.write("H4 SUPPORTED: Substantial authority-level heterogeneity\n")
                f.write(f"  → {variance_results.pct_authority:.1f}% of fine variance attributable to "
                        f"'which DPA you get'\n")
            else:
                f.write("H4 NOT SUPPORTED: Limited authority-level heterogeneity\n")

            if variance_results.icc_country > 0.1:
                f.write(f"  → {variance_results.pct_country:.1f}% of variance due to country-level policy\n")
        else:
            f.write("Variance decomposition not computed.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Generated by 8_cross_border_analysis.py (Phase 4)\n")

    logger.info(f"Saved results summary to {summary_path}")


def main() -> None:
    """Execute Phase 4: Cross-Border Analysis."""
    logger.info("=" * 70)
    logger.info("PHASE 4: CROSS-BORDER ANALYSIS")
    logger.info("=" * 70)

    # Set random seed
    np.random.seed(RANDOM_SEED)

    # Ensure output directories exist
    ensure_output_dirs()

    # Load data
    logger.info("\n[1/8] Loading data...")
    analysis_df = load_analysis_sample()

    # Prepare matching data
    logger.info("\n[2/8] Preparing matching data...")
    match_df = prepare_matching_data(analysis_df)

    # Run cross-border matching
    logger.info("\n[3/8] Running cross-border matching...")
    matching_results = run_cross_border_matching(match_df)

    # Aggregate matched pairs
    logger.info("\n[4/8] Aggregating matched pairs...")
    matched_pairs_df = aggregate_matched_pairs(matching_results)

    if len(matched_pairs_df) > 0:
        pairs_path = DATA_DIR / "matched_pairs.csv"
        matched_pairs_df.to_csv(pairs_path, index=False)
        logger.info(f"  Saved {len(matched_pairs_df)} matched pairs to {pairs_path}")

    # Statistical tests for cross-border disparities
    logger.info("\n[5/8] Testing cross-border disparities...")
    test_results = test_cross_border_disparities(matched_pairs_df)

    # Variance decomposition
    logger.info("\n[6/8] Running variance decomposition...")
    variance_results = run_variance_decomposition(match_df)

    # Generate tables
    logger.info("\n[7/8] Generating output tables...")

    # Table 6: Matching results by cohort
    table6 = format_table6_matching_results(matching_results)
    table6_path = TABLES_DIR / "table6_cross_border_matching.csv"
    table6.to_csv(table6_path, index=False)
    logger.info(f"  Saved Table 6 to {table6_path}")

    print("\n" + "=" * 60)
    print("TABLE 6: Cross-Border Matched Pairs by Article Cohort")
    print("=" * 60)
    print(table6.head(15).to_string(index=False))
    if len(table6) > 15:
        print(f"... (showing top 15 of {len(table6) - 1} cohorts)")

    # Table 7: Variance decomposition
    if variance_results:
        table7 = format_table7_variance_decomposition(variance_results)
        table7_path = TABLES_DIR / "table7_variance_decomposition.csv"
        table7.to_csv(table7_path, index=False)
        logger.info(f"  Saved Table 7 to {table7_path}")

        print("\n" + "=" * 60)
        print("TABLE 7: Variance Decomposition")
        print("=" * 60)
        print(table7.to_string(index=False))

    # Generate figures
    logger.info("\n[8/8] Creating figures...")

    # Figure 4: Matched pairs distribution
    fig4 = create_figure4_matched_pairs_distribution(matched_pairs_df, test_results)
    if fig4:
        fig4_path = FIGURES_DIR / "figure4_matched_pairs_distribution.png"
        fig4.savefig(fig4_path, dpi=300, bbox_inches="tight")
        plt.close(fig4)
        logger.info(f"  Saved Figure 4 to {fig4_path}")

        # PDF version
        fig4_pdf = FIGURES_DIR / "figure4_matched_pairs_distribution.pdf"
        fig4 = create_figure4_matched_pairs_distribution(matched_pairs_df, test_results)
        if fig4:
            fig4.savefig(fig4_pdf, format="pdf", bbox_inches="tight")
            plt.close(fig4)

    # Figure 5: Variance partition
    if variance_results:
        fig5 = create_figure5_variance_partition(variance_results)
        if fig5:
            fig5_path = FIGURES_DIR / "figure5_variance_partition.png"
            fig5.savefig(fig5_path, dpi=300, bbox_inches="tight")
            plt.close(fig5)
            logger.info(f"  Saved Figure 5 to {fig5_path}")

            # PDF version
            fig5_pdf = FIGURES_DIR / "figure5_variance_partition.pdf"
            fig5 = create_figure5_variance_partition(variance_results)
            if fig5:
                fig5.savefig(fig5_pdf, format="pdf", bbox_inches="tight")
                plt.close(fig5)

    # Save summary
    save_phase4_summary(matching_results, test_results, variance_results)

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4 COMPLETE")
    logger.info("=" * 70)

    print("\n" + "=" * 60)
    print("PHASE 4 SUMMARY: CROSS-BORDER ANALYSIS")
    print("=" * 60)

    total_pairs = sum(r.n_matched_pairs for r in matching_results)
    print(f"\nCross-Border Matching:")
    print(f"  Cohorts analyzed: {len(matching_results)}")
    print(f"  Total matched pairs: {total_pairs}")

    if test_results:
        print(f"\nH3 Test (Cross-Border Disparities):")
        print(f"  Mean Δ(log fine): {test_results.get('mean_delta', 0):.4f}")
        print(f"  t-statistic: {test_results.get('t_stat', 0):.3f}")
        print(f"  p-value: {test_results.get('t_pvalue_onesided', 1):.4f}")
        sig = "SUPPORTED" if test_results.get('t_pvalue_onesided', 1) < 0.05 else "NOT SUPPORTED"
        print(f"  H3 (disparities exist): {sig}")

    if variance_results:
        print(f"\nVariance Decomposition (H4):")
        print(f"  Country-level: {variance_results.pct_country:.1f}%")
        print(f"  Authority-level: {variance_results.pct_authority:.1f}%")
        print(f"  Case-level: {variance_results.pct_residual:.1f}%")
        print(f"  Combined ICC: {variance_results.icc_combined:.4f}")
        sig = "SUPPORTED" if variance_results.icc_authority > 0.1 else "NOT SUPPORTED"
        print(f"  H4 (authority heterogeneity): {sig}")

    print("\n" + "=" * 60)
    print(f"Outputs saved to:")
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Data: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
