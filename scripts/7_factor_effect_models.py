"""Phase 3 – Factor Effect Models: Mixed-Effects Regression Analysis.

This script implements Phase 3 (Factor Effect Models) of the methodology proposal:
"Reasoning and Consistency in GDPR Enforcement: A Cross-Border Analysis"

Models Implemented:
1. Model 1: Aggregate factor effects on log fine magnitude (Mixed-effects)
2. Model 2: Factor-by-factor decomposition (each Art 83(2) factor separately)
3. Model 3: Systematicity → Predictability regression (Authority-level OLS)

Outputs:
- Table 3: Model 1 main results (factor effects on log fine)
- Table 4: Factor-by-factor decomposition coefficients
- Table 5: Authority systematicity rankings with predictability
- Figure 3: Systematicity vs. Fine Predictability scatter plot

Input:  outputs/paper/data/analysis_sample.csv
        outputs/paper/data/authority_systematicity.csv
Output: outputs/paper/tables/
        outputs/paper/figures/
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

# Suppress convergence warnings for display
warnings.filterwarnings('ignore', category=RuntimeWarning)

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
SYSTEMATICITY_PATH = Path("outputs/paper/data/authority_systematicity.csv")

# Output directories
TABLES_DIR = Path("outputs/paper/tables")
FIGURES_DIR = Path("outputs/paper/figures")
DATA_DIR = Path("outputs/paper/data")

# Random seed for reproducibility
RANDOM_SEED = 42

# Minimum observations for authority-level analysis
MIN_AUTHORITY_OBS = 10

# Article 83(2) factor columns and their score columns
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

ART83_SCORE_COLS = [f"{col}_score" for col in ART83_FACTOR_COLS]

# Factor labels for display
FACTOR_LABELS = {
    "a59_nature_gravity_duration_score": "Nature/Gravity/Duration",
    "a60_intentional_negligent_score": "Intentional/Negligent",
    "a61_mitigate_damage_actions_score": "Mitigation Actions",
    "a62_technical_org_measures_score": "Tech/Org Measures",
    "a63_previous_infringements_score": "Previous Infringements",
    "a64_cooperation_authority_score": "Cooperation",
    "a65_data_categories_affected_score": "Data Categories",
    "a66_infringement_became_known_score": "How Infringement Known",
    "a67_prior_orders_compliance_score": "Prior Order Compliance",
    "a68_codes_certification_score": "Codes/Certification",
    "a69_other_factors_score": "Other Factors",
}

# Processing context columns
CONTEXT_COLS = [
    "has_cctv", "has_marketing", "has_cookies", "has_employee_monitoring",
    "has_ai", "has_problematic_third_party_sharing", "has_health_context",
]

# Violation indicator columns
VIOLATION_COLS = [
    "breach_has_art5", "breach_has_art6", "breach_has_art13",
    "breach_has_art32", "breach_has_art33",
]

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
            "Run Phase 1 (6_paper_data_preparation.py) first."
        )
    df = pd.read_csv(ANALYSIS_SAMPLE_PATH, low_memory=False)
    logger.info(f"Loaded analysis sample: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_systematicity() -> pd.DataFrame:
    """Load authority systematicity indices from Phase 1."""
    if not SYSTEMATICITY_PATH.exists():
        raise FileNotFoundError(
            f"Systematicity indices not found at {SYSTEMATICITY_PATH}. "
            "Run Phase 1 (6_paper_data_preparation.py) first."
        )
    df = pd.read_csv(SYSTEMATICITY_PATH)
    logger.info(f"Loaded systematicity indices: {len(df)} authorities")
    return df


def prepare_regression_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for regression analysis.

    - Ensure required columns exist
    - Create dummy variables for categorical controls
    - Handle missing values appropriately
    """
    data = df.copy()

    # Ensure log_fine_2025 exists
    if "log_fine_2025" not in data.columns:
        fine_col = "fine_amount_eur_real_2025" if "fine_amount_eur_real_2025" in data.columns else "fine_amount_eur"
        data["log_fine_2025"] = np.log1p(data[fine_col].clip(lower=0).fillna(0))

    # Create authority identifier
    if "authority_name_norm" not in data.columns:
        data["authority_name_norm"] = data["a2_authority_name"]

    # Fill missing factor scores with 0 (NOT_DISCUSSED = neutral)
    for col in ART83_SCORE_COLS:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    # Fill aggregate factor columns
    for col in ["art83_aggravating_count", "art83_mitigating_count",
                "art83_neutral_count", "art83_discussed_count", "art83_balance_score"]:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    # Create dummy variables for defendant class
    if "a8_defendant_class" in data.columns:
        data["is_private"] = (data["a8_defendant_class"] == "PRIVATE").astype(int)
        data["is_public"] = (data["a8_defendant_class"] == "PUBLIC").astype(int)

    # Create dummy for enterprise size
    if "a9_enterprise_size" in data.columns:
        data["is_large_enterprise"] = data["a9_enterprise_size"].isin(
            ["LARGE", "VERY_LARGE"]
        ).astype(int)
        data["is_sme"] = (data["a9_enterprise_size"] == "SME").astype(int)

    # Create year dummies for time effects
    if "decision_year" in data.columns:
        data["year_2019"] = (data["decision_year"] == 2019).astype(int)
        data["year_2020"] = (data["decision_year"] == 2020).astype(int)
        data["year_2021"] = (data["decision_year"] == 2021).astype(int)
        data["year_2022"] = (data["decision_year"] == 2022).astype(int)
        data["year_2023"] = (data["decision_year"] == 2023).astype(int)
        data["year_2024"] = (data["decision_year"] == 2024).astype(int)

    # Fill context columns
    for col in CONTEXT_COLS:
        if col in data.columns:
            data[col] = data[col].fillna(False).astype(int)

    # Fill violation columns
    for col in VIOLATION_COLS:
        if col in data.columns:
            data[col] = data[col].fillna(False).astype(int)

    # Complaint and audit origin
    if "has_complaint_bool" in data.columns:
        data["complaint_origin"] = data["has_complaint_bool"].fillna(False)
        data["complaint_origin"] = data["complaint_origin"].astype(bool).astype(int)
    if "official_audit_bool" in data.columns:
        data["audit_origin"] = data["official_audit_bool"].fillna(False)
        data["audit_origin"] = data["audit_origin"].astype(bool).astype(int)

    # Cross-border (OSS) flag
    if "oss_case_bool" in data.columns:
        data["oss_case"] = data["oss_case_bool"].fillna(False)
        data["oss_case"] = data["oss_case"].astype(bool).astype(int)

    # Encode authority as categorical for mixed effects
    data["authority_id"] = pd.Categorical(data["authority_name_norm"]).codes

    logger.info(f"Prepared regression data: {len(data)} observations")
    return data


# -----------------------------------------------------------------------------
# Model 1: Aggregate Factor Effects (Mixed-Effects)
# -----------------------------------------------------------------------------

@dataclass
class Model1Results:
    """Container for Model 1 (aggregate factors) results."""
    n_obs: int
    n_groups: int
    coefficients: Dict[str, float]
    std_errors: Dict[str, float]
    p_values: Dict[str, float]
    conf_intervals: Dict[str, Tuple[float, float]]
    icc: float  # Intraclass correlation
    var_authority: float
    var_residual: float
    log_likelihood: float
    aic: float
    bic: float
    r2_marginal: float  # Fixed effects only
    r2_conditional: float  # Fixed + random


def run_model1_aggregate_factors(df: pd.DataFrame) -> Optional[Model1Results]:
    """
    Run Model 1: Aggregate factor effects on log fine magnitude.

    Specification:
    log_fine_2025 = β₀ + β₁·art83_aggravating_count
                       + β₂·art83_mitigating_count
                       + β₃·art83_neutral_count
                       + γ·X (contexts) + δ·V (violations) + θ·D (defendant)
                       + u_a (authority random effect) + ε
    """
    if not HAS_STATSMODELS:
        logger.error("statsmodels required for Model 1")
        return None

    # Define formula components
    factor_vars = ["art83_aggravating_count", "art83_mitigating_count", "art83_neutral_count"]
    context_vars = [c for c in CONTEXT_COLS if c in df.columns and df[c].var() > 0]
    violation_vars = [v for v in VIOLATION_COLS if v in df.columns and df[v].var() > 0]
    defendant_vars = ["is_private", "is_large_enterprise"]
    year_vars = ["year_2020", "year_2021", "year_2022", "year_2023", "year_2024"]
    procedural_vars = ["complaint_origin", "audit_origin", "oss_case"]

    # Build covariate list
    all_covariates = factor_vars.copy()
    all_covariates += [v for v in context_vars if v in df.columns]
    all_covariates += [v for v in violation_vars if v in df.columns]
    all_covariates += [v for v in defendant_vars if v in df.columns]
    all_covariates += [v for v in year_vars if v in df.columns]
    all_covariates += [v for v in procedural_vars if v in df.columns]

    # Filter to complete cases
    keep_cols = ["log_fine_2025", "authority_name_norm"] + all_covariates
    keep_cols = [c for c in keep_cols if c in df.columns]
    model_df = df[keep_cols].dropna()

    logger.info(f"Model 1 sample: {len(model_df)} complete observations")

    if len(model_df) < 50:
        logger.warning("Insufficient observations for Model 1")
        return None

    # Build formula
    formula = "log_fine_2025 ~ " + " + ".join(all_covariates)

    try:
        # Fit mixed-effects model with authority random intercept
        model = smf.mixedlm(
            formula,
            data=model_df,
            groups=model_df["authority_name_norm"],
            re_formula="~1"  # Random intercept only
        )
        result = model.fit(method="powell", maxiter=500)

        # Extract results
        coef_dict = dict(result.fe_params)
        se_dict = dict(result.bse_fe)
        pval_dict = dict(result.pvalues)

        # Confidence intervals
        ci = result.conf_int()
        ci_dict = {k: (ci.loc[k, 0], ci.loc[k, 1]) for k in coef_dict.keys()}

        # Variance components
        var_auth = result.cov_re.iloc[0, 0] if hasattr(result.cov_re, 'iloc') else float(result.cov_re)
        var_resid = result.scale

        # ICC = var_authority / (var_authority + var_residual)
        icc = var_auth / (var_auth + var_resid) if (var_auth + var_resid) > 0 else 0

        # Pseudo R² calculations
        # Marginal R² (fixed effects only) - using fitted values
        y = model_df["log_fine_2025"].values
        fitted_vals = result.fittedvalues
        ss_total = np.var(y)
        ss_fixed = np.var(fitted_vals)
        r2_marginal = ss_fixed / ss_total if ss_total > 0 else 0

        # Conditional R² (fixed + random) - based on residuals
        residuals = y - fitted_vals
        ss_resid = np.var(residuals)
        r2_conditional = 1 - (ss_resid / ss_total) if ss_total > 0 else 0

        return Model1Results(
            n_obs=int(result.nobs),
            n_groups=int(result.nobs / result.nobs) if hasattr(result, 'k_re') else model_df["authority_name_norm"].nunique(),
            coefficients=coef_dict,
            std_errors=se_dict,
            p_values=pval_dict,
            conf_intervals=ci_dict,
            icc=icc,
            var_authority=var_auth,
            var_residual=var_resid,
            log_likelihood=result.llf,
            aic=result.aic,
            bic=result.bic,
            r2_marginal=r2_marginal,
            r2_conditional=r2_conditional,
        )

    except Exception as e:
        logger.error(f"Model 1 estimation failed: {e}")
        return None


def format_model1_table(results: Model1Results) -> pd.DataFrame:
    """Format Model 1 results as publication-ready Table 3."""
    rows = []

    # Key coefficient order for presentation
    key_vars = [
        ("art83_aggravating_count", "Aggravating Factor Count"),
        ("art83_mitigating_count", "Mitigating Factor Count"),
        ("art83_neutral_count", "Neutral Factor Count"),
        ("is_private", "Private Sector"),
        ("is_large_enterprise", "Large/Very Large Enterprise"),
        ("complaint_origin", "Complaint Origin"),
        ("audit_origin", "Audit Origin"),
        ("oss_case", "Cross-Border (OSS) Case"),
        ("breach_has_art5", "Art. 5 Violation"),
        ("breach_has_art6", "Art. 6 Violation"),
        ("breach_has_art32", "Art. 32 Violation"),
        ("breach_has_art33", "Art. 33 Violation"),
        ("has_cctv", "CCTV Context"),
        ("has_marketing", "Marketing Context"),
        ("has_health_context", "Health Data Context"),
        ("Intercept", "Intercept"),
    ]

    for var, label in key_vars:
        if var in results.coefficients:
            coef = results.coefficients[var]
            se = results.std_errors.get(var, np.nan)
            pval = results.p_values.get(var, np.nan)
            ci = results.conf_intervals.get(var, (np.nan, np.nan))

            # Significance stars
            stars = ""
            if pval < 0.001:
                stars = "***"
            elif pval < 0.01:
                stars = "**"
            elif pval < 0.05:
                stars = "*"
            elif pval < 0.1:
                stars = "†"

            rows.append({
                "Variable": label,
                "Coefficient": f"{coef:.4f}{stars}",
                "Std. Error": f"{se:.4f}",
                "95% CI": f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                "p-value": f"{pval:.4f}" if pval >= 0.0001 else "<0.0001",
            })

    # Add model diagnostics
    rows.append({"Variable": "---", "Coefficient": "---", "Std. Error": "---", "95% CI": "---", "p-value": "---"})
    rows.append({"Variable": "N (observations)", "Coefficient": str(results.n_obs), "Std. Error": "", "95% CI": "", "p-value": ""})
    rows.append({"Variable": "N (authorities)", "Coefficient": str(results.n_groups), "Std. Error": "", "95% CI": "", "p-value": ""})
    rows.append({"Variable": "σ²_authority", "Coefficient": f"{results.var_authority:.4f}", "Std. Error": "", "95% CI": "", "p-value": ""})
    rows.append({"Variable": "σ²_residual", "Coefficient": f"{results.var_residual:.4f}", "Std. Error": "", "95% CI": "", "p-value": ""})
    rows.append({"Variable": "ICC (authority)", "Coefficient": f"{results.icc:.4f}", "Std. Error": "", "95% CI": "", "p-value": ""})
    rows.append({"Variable": "R² (marginal)", "Coefficient": f"{results.r2_marginal:.4f}", "Std. Error": "", "95% CI": "", "p-value": ""})
    rows.append({"Variable": "R² (conditional)", "Coefficient": f"{results.r2_conditional:.4f}", "Std. Error": "", "95% CI": "", "p-value": ""})
    rows.append({"Variable": "AIC", "Coefficient": f"{results.aic:.2f}", "Std. Error": "", "95% CI": "", "p-value": ""})
    rows.append({"Variable": "Log-Likelihood", "Coefficient": f"{results.log_likelihood:.2f}", "Std. Error": "", "95% CI": "", "p-value": ""})

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Model 2: Factor-by-Factor Decomposition
# -----------------------------------------------------------------------------

@dataclass
class FactorCoefficient:
    """Container for individual factor coefficient."""
    factor_name: str
    factor_label: str
    coefficient: float
    std_error: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_nonmissing: int


def run_model2_factor_decomposition(df: pd.DataFrame) -> List[FactorCoefficient]:
    """
    Run Model 2: Factor-by-factor decomposition.

    Estimates the effect of each Article 83(2) factor separately,
    controlling for other covariates.

    Specification for each factor j:
    log_fine_2025 = β₀ + βⱼ·factor_j_score + γ·X + δ·V + θ·D + u_a + ε
    """
    if not HAS_STATSMODELS:
        logger.error("statsmodels required for Model 2")
        return []

    results: List[FactorCoefficient] = []

    # Control variables (excluding other factors)
    context_vars = [c for c in CONTEXT_COLS if c in df.columns and df[c].var() > 0]
    violation_vars = [v for v in VIOLATION_COLS if v in df.columns and df[v].var() > 0]
    control_vars = ["is_private", "is_large_enterprise", "complaint_origin",
                    "audit_origin", "oss_case", "year_2020", "year_2021",
                    "year_2022", "year_2023", "year_2024"]
    control_vars = [c for c in control_vars if c in df.columns]
    control_vars += context_vars + violation_vars

    for score_col in ART83_SCORE_COLS:
        if score_col not in df.columns:
            continue

        # Filter to cases where this factor was discussed (non-zero variance)
        model_df = df[df[score_col].notna()].copy()

        # Check for variation
        if model_df[score_col].var() == 0 or len(model_df) < 30:
            logger.info(f"Skipping {score_col}: insufficient variation or observations")
            continue

        # Build formula with this factor
        available_controls = [c for c in control_vars if c in model_df.columns and model_df[c].var() > 0]
        formula = f"log_fine_2025 ~ {score_col}"
        if available_controls:
            formula += " + " + " + ".join(available_controls)

        try:
            # Use mixed-effects model with authority random effect
            keep_cols = ["log_fine_2025", "authority_name_norm", score_col] + available_controls
            reg_df = model_df[keep_cols].dropna()

            if len(reg_df) < 30:
                continue

            model = smf.mixedlm(
                formula,
                data=reg_df,
                groups=reg_df["authority_name_norm"],
                re_formula="~1"
            )
            fit = model.fit(method="powell", maxiter=300)

            # Extract factor coefficient
            coef = fit.fe_params[score_col]
            se = fit.bse_fe[score_col]
            pval = fit.pvalues[score_col]
            ci = fit.conf_int().loc[score_col]
            t_stat = coef / se if se > 0 else np.nan

            results.append(FactorCoefficient(
                factor_name=score_col,
                factor_label=FACTOR_LABELS.get(score_col, score_col),
                coefficient=coef,
                std_error=se,
                t_stat=t_stat,
                p_value=pval,
                ci_lower=ci[0],
                ci_upper=ci[1],
                n_nonmissing=len(reg_df),
            ))

            logger.info(f"  {FACTOR_LABELS.get(score_col, score_col)}: β={coef:.4f}, p={pval:.4f}")

        except Exception as e:
            logger.warning(f"Model 2 failed for {score_col}: {e}")
            continue

    return results


def format_model2_table(results: List[FactorCoefficient]) -> pd.DataFrame:
    """Format Model 2 results as publication-ready Table 4."""
    rows = []

    for r in sorted(results, key=lambda x: abs(x.coefficient), reverse=True):
        # Significance stars
        stars = ""
        if r.p_value < 0.001:
            stars = "***"
        elif r.p_value < 0.01:
            stars = "**"
        elif r.p_value < 0.05:
            stars = "*"
        elif r.p_value < 0.1:
            stars = "†"

        rows.append({
            "Article 83(2) Factor": r.factor_label,
            "Coefficient": f"{r.coefficient:.4f}{stars}",
            "Std. Error": f"{r.std_error:.4f}",
            "t-statistic": f"{r.t_stat:.3f}",
            "p-value": f"{r.p_value:.4f}" if r.p_value >= 0.0001 else "<0.0001",
            "95% CI": f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]",
            "N": r.n_nonmissing,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Model 3: Systematicity → Predictability Analysis
# -----------------------------------------------------------------------------

@dataclass
class AuthorityPredictability:
    """Container for authority-level predictability metrics."""
    authority_id: str
    n_decisions: int
    r2_baseline: float  # R² from baseline model within authority
    rmse: float  # Root mean squared error
    mean_abs_residual: float


@dataclass
class Model3Results:
    """Container for Model 3 (systematicity-predictability) results."""
    n_authorities: int
    beta_systematicity: float
    se_systematicity: float
    t_stat: float
    p_value: float
    r2: float
    correlation: float
    authority_predictability: List[AuthorityPredictability]


def compute_authority_predictability(df: pd.DataFrame) -> List[AuthorityPredictability]:
    """
    Compute fine predictability (R²) for each authority.

    For each authority, estimate baseline regression:
    log_fine_2025 ~ art83_balance_score + controls

    Returns R² and RMSE for each authority with sufficient data.
    """
    results: List[AuthorityPredictability] = []

    for authority, group in df.groupby("authority_name_norm"):
        if pd.isna(authority) or len(group) < MIN_AUTHORITY_OBS:
            continue

        # Simple OLS within authority
        y = group["log_fine_2025"].values
        X_cols = ["art83_balance_score"]

        # Add controls if available
        for col in ["is_private", "is_large_enterprise", "breach_has_art5", "breach_has_art6"]:
            if col in group.columns and group[col].var() > 0:
                X_cols.append(col)

        X = group[X_cols].fillna(0).values

        if len(X) < len(X_cols) + 2:  # Need more obs than parameters
            continue

        try:
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const)
            fit = model.fit()

            r2 = fit.rsquared
            residuals = fit.resid
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))

            results.append(AuthorityPredictability(
                authority_id=str(authority),
                n_decisions=len(group),
                r2_baseline=r2,
                rmse=rmse,
                mean_abs_residual=mae,
            ))
        except Exception:
            continue

    return results


def run_model3_systematicity_predictability(
    systematicity_df: pd.DataFrame,
    predictability_list: List[AuthorityPredictability]
) -> Optional[Model3Results]:
    """
    Run Model 3: Systematicity → Predictability regression.

    Specification:
    R²_a = α + β·Systematicity_a + ε_a

    Tests H2: β > 0 indicates systematic reasoning improves predictability.
    """
    if not HAS_STATSMODELS:
        logger.error("statsmodels required for Model 3")
        return None

    # Create merged dataset
    pred_df = pd.DataFrame([
        {"authority_id": p.authority_id, "r2_baseline": p.r2_baseline,
         "rmse": p.rmse, "n_decisions": p.n_decisions}
        for p in predictability_list
    ])

    merged = pred_df.merge(systematicity_df, on="authority_id", how="inner")

    if len(merged) < 5:
        logger.warning("Insufficient authorities for Model 3 (need at least 5)")
        return None

    logger.info(f"Model 3 sample: {len(merged)} authorities")

    # OLS regression: R² ~ Systematicity
    y = merged["r2_baseline"].values
    X = merged["systematicity"].values
    X_const = sm.add_constant(X)

    try:
        model = sm.OLS(y, X_const)
        fit = model.fit()

        beta = fit.params[1]
        se = fit.bse[1]
        t_stat = fit.tvalues[1]
        p_val = fit.pvalues[1]
        r2 = fit.rsquared

        # Pearson correlation
        corr, _ = stats.pearsonr(merged["systematicity"], merged["r2_baseline"])

        return Model3Results(
            n_authorities=len(merged),
            beta_systematicity=beta,
            se_systematicity=se,
            t_stat=t_stat,
            p_value=p_val,
            r2=r2,
            correlation=corr,
            authority_predictability=predictability_list,
        )

    except Exception as e:
        logger.error(f"Model 3 failed: {e}")
        return None


def format_model3_table(
    systematicity_df: pd.DataFrame,
    predictability_list: List[AuthorityPredictability],
    model3_results: Optional[Model3Results]
) -> pd.DataFrame:
    """Format Table 5: Authority systematicity rankings with predictability."""
    # Merge data
    pred_dict = {p.authority_id: p for p in predictability_list}

    rows = []
    for _, row in systematicity_df.iterrows():
        auth_id = row["authority_id"]
        pred = pred_dict.get(auth_id)

        rows.append({
            "Rank": 0,  # Will fill after sorting
            "Authority": auth_id[:45] + "..." if len(auth_id) > 48 else auth_id,
            "N Decisions": int(row["n_decisions"]),
            "Coverage": f"{row['coverage']:.3f}",
            "Consistency": f"{row['consistency']:.3f}",
            "Coherence": f"{row['coherence']:.3f}",
            "Systematicity": f"{row['systematicity']:.4f}",
            "R² (Predictability)": f"{pred.r2_baseline:.3f}" if pred else "—",
            "RMSE": f"{pred.rmse:.3f}" if pred else "—",
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Systematicity", ascending=False)
    df["Rank"] = range(1, len(df) + 1)

    return df


def create_figure3_scatter(
    systematicity_df: pd.DataFrame,
    predictability_list: List[AuthorityPredictability],
    model3_results: Optional[Model3Results]
) -> Optional[plt.Figure]:
    """Create Figure 3: Systematicity vs. Fine Predictability scatter."""
    if not HAS_PLOTTING:
        logger.warning("Plotting libraries not available")
        return None

    # Merge data
    pred_df = pd.DataFrame([
        {"authority_id": p.authority_id, "r2_baseline": p.r2_baseline,
         "n_decisions": p.n_decisions}
        for p in predictability_list
    ])
    merged = pred_df.merge(systematicity_df, on="authority_id", how="inner")

    if len(merged) < 3:
        logger.warning("Insufficient data for scatter plot")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot with size proportional to n_decisions
    sizes = merged["n_decisions_x"] / merged["n_decisions_x"].max() * 300 + 50
    scatter = ax.scatter(
        merged["systematicity"],
        merged["r2_baseline"],
        s=sizes,
        alpha=0.6,
        c=merged["coverage"],
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5
    )

    # Add regression line if Model 3 results available
    if model3_results and model3_results.p_value < 0.1:
        x_line = np.linspace(merged["systematicity"].min(), merged["systematicity"].max(), 100)
        y_line = model3_results.beta_systematicity * x_line + (
            merged["r2_baseline"].mean() - model3_results.beta_systematicity * merged["systematicity"].mean()
        )
        ax.plot(x_line, y_line, 'r--', linewidth=2,
                label=f"β = {model3_results.beta_systematicity:.3f} (p = {model3_results.p_value:.3f})")
        ax.legend(loc="upper left")

    # Labels and formatting
    ax.set_xlabel("Systematicity Index", fontsize=12)
    ax.set_ylabel("Fine Predictability (R²)", fontsize=12)
    ax.set_title("Authority Systematicity vs. Fine Predictability\n(bubble size = n decisions, color = coverage)",
                 fontsize=14)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Factor Coverage", fontsize=10)

    # Add authority labels for outliers
    for _, row in merged.iterrows():
        if row["systematicity"] > merged["systematicity"].quantile(0.9) or \
           row["r2_baseline"] > merged["r2_baseline"].quantile(0.9):
            ax.annotate(
                row["authority_id"][:20],
                (row["systematicity"], row["r2_baseline"]),
                fontsize=7,
                alpha=0.7
            )

    # Grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Output Generation
# -----------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    """Create output directories if they don't exist."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directories ready: {TABLES_DIR}, {FIGURES_DIR}")


def save_results_summary(
    model1: Optional[Model1Results],
    model2: List[FactorCoefficient],
    model3: Optional[Model3Results]
) -> None:
    """Save summary of all Phase 3 results."""
    summary_path = DATA_DIR / "phase3_results_summary.txt"

    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 3: FACTOR EFFECT MODELS - RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        # Model 1 Summary
        f.write("MODEL 1: AGGREGATE FACTOR EFFECTS\n")
        f.write("-" * 40 + "\n")
        if model1:
            f.write(f"Observations: {model1.n_obs}\n")
            f.write(f"Authorities (groups): {model1.n_groups}\n")
            f.write(f"\nKey Coefficients:\n")
            for var in ["art83_aggravating_count", "art83_mitigating_count", "art83_neutral_count"]:
                if var in model1.coefficients:
                    coef = model1.coefficients[var]
                    pval = model1.p_values[var]
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                    f.write(f"  {var}: {coef:.4f}{sig} (p={pval:.4f})\n")
            f.write(f"\nVariance Decomposition:\n")
            f.write(f"  σ²_authority: {model1.var_authority:.4f}\n")
            f.write(f"  σ²_residual: {model1.var_residual:.4f}\n")
            f.write(f"  ICC: {model1.icc:.4f} ({model1.icc*100:.1f}% of variance at authority level)\n")
            f.write(f"\nModel Fit:\n")
            f.write(f"  R² (marginal): {model1.r2_marginal:.4f}\n")
            f.write(f"  R² (conditional): {model1.r2_conditional:.4f}\n")
            f.write(f"  AIC: {model1.aic:.2f}\n")
        else:
            f.write("Model 1 estimation failed.\n")

        f.write("\n")

        # Model 2 Summary
        f.write("MODEL 2: FACTOR-BY-FACTOR DECOMPOSITION\n")
        f.write("-" * 40 + "\n")
        if model2:
            # Sort by absolute coefficient
            sorted_factors = sorted(model2, key=lambda x: abs(x.coefficient), reverse=True)
            f.write("Factor coefficients (sorted by magnitude):\n\n")
            for r in sorted_factors:
                sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
                f.write(f"  {r.factor_label:30s}: β={r.coefficient:+.4f}{sig:3s} (SE={r.std_error:.4f}, N={r.n_nonmissing})\n")
        else:
            f.write("No factor decomposition results.\n")

        f.write("\n")

        # Model 3 Summary
        f.write("MODEL 3: SYSTEMATICITY → PREDICTABILITY\n")
        f.write("-" * 40 + "\n")
        if model3:
            f.write(f"Authorities analyzed: {model3.n_authorities}\n")
            sig = "***" if model3.p_value < 0.001 else "**" if model3.p_value < 0.01 else "*" if model3.p_value < 0.05 else ""
            f.write(f"\nRegression: R²_a = α + β·Systematicity_a\n")
            f.write(f"  β_systematicity: {model3.beta_systematicity:.4f}{sig} (SE={model3.se_systematicity:.4f})\n")
            f.write(f"  t-statistic: {model3.t_stat:.3f}\n")
            f.write(f"  p-value: {model3.p_value:.4f}\n")
            f.write(f"  R²: {model3.r2:.4f}\n")
            f.write(f"  Correlation (Pearson): {model3.correlation:.4f}\n")
            f.write(f"\nInterpretation: ")
            if model3.p_value < 0.05:
                if model3.beta_systematicity > 0:
                    f.write("Higher systematicity → Higher predictability (H2 SUPPORTED)\n")
                else:
                    f.write("Higher systematicity → Lower predictability (UNEXPECTED)\n")
            else:
                f.write("No significant relationship between systematicity and predictability\n")
        else:
            f.write("Model 3 estimation failed.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Generated by 7_factor_effect_models.py (Phase 3)\n")

    logger.info(f"Saved results summary to {summary_path}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main() -> None:
    """Execute Phase 3: Factor Effect Models."""
    logger.info("=" * 70)
    logger.info("PHASE 3: FACTOR EFFECT MODELS")
    logger.info("=" * 70)

    # Set random seed
    np.random.seed(RANDOM_SEED)

    # Ensure output directories exist
    ensure_output_dirs()

    # Load data
    logger.info("\n[1/7] Loading data...")
    analysis_df = load_analysis_sample()
    systematicity_df = load_systematicity()

    # Prepare regression data
    logger.info("\n[2/7] Preparing regression data...")
    reg_df = prepare_regression_data(analysis_df)

    # Model 1: Aggregate Factor Effects
    logger.info("\n[3/7] Running Model 1: Aggregate Factor Effects...")
    model1_results = run_model1_aggregate_factors(reg_df)

    if model1_results:
        table3 = format_model1_table(model1_results)
        table3_path = TABLES_DIR / "table3_aggregate_factors.csv"
        table3.to_csv(table3_path, index=False)
        logger.info(f"  Saved Table 3 to {table3_path}")
        print("\n" + "=" * 50)
        print("TABLE 3: Aggregate Factor Effects on Log Fine")
        print("=" * 50)
        print(table3.to_string(index=False))

    # Model 2: Factor-by-Factor Decomposition
    logger.info("\n[4/7] Running Model 2: Factor Decomposition...")
    model2_results = run_model2_factor_decomposition(reg_df)

    if model2_results:
        table4 = format_model2_table(model2_results)
        table4_path = TABLES_DIR / "table4_factor_decomposition.csv"
        table4.to_csv(table4_path, index=False)
        logger.info(f"  Saved Table 4 to {table4_path}")
        print("\n" + "=" * 50)
        print("TABLE 4: Factor-by-Factor Decomposition")
        print("=" * 50)
        print(table4.to_string(index=False))

    # Compute authority-level predictability
    logger.info("\n[5/7] Computing authority predictability...")
    predictability_list = compute_authority_predictability(reg_df)
    logger.info(f"  Computed predictability for {len(predictability_list)} authorities")

    # Model 3: Systematicity → Predictability
    logger.info("\n[6/7] Running Model 3: Systematicity-Predictability Analysis...")
    model3_results = run_model3_systematicity_predictability(systematicity_df, predictability_list)

    # Table 5: Authority Rankings
    table5 = format_model3_table(systematicity_df, predictability_list, model3_results)
    table5_path = TABLES_DIR / "table5_authority_systematicity.csv"
    table5.to_csv(table5_path, index=False)
    logger.info(f"  Saved Table 5 to {table5_path}")
    print("\n" + "=" * 50)
    print("TABLE 5: Authority Systematicity Rankings")
    print("=" * 50)
    print(table5.head(10).to_string(index=False))
    print("... (showing top 10)")

    # Figure 3: Scatter Plot
    logger.info("\n[7/7] Creating Figure 3: Systematicity vs Predictability...")
    fig3 = create_figure3_scatter(systematicity_df, predictability_list, model3_results)
    if fig3:
        fig3_path = FIGURES_DIR / "figure3_systematicity_predictability.png"
        fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
        plt.close(fig3)
        logger.info(f"  Saved Figure 3 to {fig3_path}")

        # Also save PDF version
        fig3_pdf = FIGURES_DIR / "figure3_systematicity_predictability.pdf"
        fig3 = create_figure3_scatter(systematicity_df, predictability_list, model3_results)
        if fig3:
            fig3.savefig(fig3_pdf, format="pdf", bbox_inches="tight")
            plt.close(fig3)

    # Save results summary
    save_results_summary(model1_results, model2_results, model3_results)

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3 COMPLETE")
    logger.info("=" * 70)

    print("\n" + "=" * 50)
    print("PHASE 3 SUMMARY")
    print("=" * 50)

    if model1_results:
        print(f"\nModel 1 (Aggregate Factors):")
        print(f"  N = {model1_results.n_obs} observations, {model1_results.n_groups} authorities")
        print(f"  Aggravating count: β = {model1_results.coefficients.get('art83_aggravating_count', 0):.4f}")
        print(f"  Mitigating count:  β = {model1_results.coefficients.get('art83_mitigating_count', 0):.4f}")
        print(f"  ICC (authority): {model1_results.icc:.4f} ({model1_results.icc*100:.1f}% variance)")

    if model2_results:
        sig_factors = [f for f in model2_results if f.p_value < 0.05]
        print(f"\nModel 2 (Factor Decomposition):")
        print(f"  Factors analyzed: {len(model2_results)}")
        print(f"  Significant (p<0.05): {len(sig_factors)}")
        if sig_factors:
            top_factor = max(sig_factors, key=lambda x: abs(x.coefficient))
            print(f"  Strongest factor: {top_factor.factor_label} (β={top_factor.coefficient:.4f})")

    if model3_results:
        print(f"\nModel 3 (Systematicity → Predictability):")
        print(f"  N = {model3_results.n_authorities} authorities")
        print(f"  β_systematicity = {model3_results.beta_systematicity:.4f} (p={model3_results.p_value:.4f})")
        print(f"  Correlation = {model3_results.correlation:.4f}")

    print("\n" + "=" * 50)
    print(f"Outputs saved to:")
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Data: {DATA_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
