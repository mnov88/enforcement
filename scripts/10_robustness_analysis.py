"""Phase 5 – Robustness & Finalization: Specification Curves, Bootstrap, Sensitivity.

This script implements Phase 5 (Robustness & Finalization) of the methodology proposal:
"Reasoning and Consistency in GDPR Enforcement: A Cross-Border Analysis"

Analyses Implemented:
1. Specification Curve Analysis (§6.1) - Model 1 across all control set combinations
2. Bootstrap Confidence Intervals (§6.5) - 1000 bootstrap replicates for key estimates
3. Leave-One-Country-Out (§6.3) - Coefficient stability across country exclusions
4. Placebo Tests (§6.4) - Random permutation of article assignments and factor scores
5. Alternative Factor Operationalizations (§6.6) - Binary counts, weighted scores, PCA

Outputs:
- Table 8: Specification curve summary
- Figure 7: Specification curve plot (aggravating factor effect across models)
- Table S1: Bootstrap confidence intervals for key estimates
- Table S2: Leave-one-country-out coefficient stability
- Table S3: Placebo test results
- Table S4: Alternative operationalization comparison

Input:  outputs/paper/data/analysis_sample.csv
        outputs/paper/tables/table3_aggregate_factors.csv
Output: outputs/paper/tables/, outputs/paper/figures/, outputs/paper/supplementary/
"""

from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

# Suppress warnings for cleaner output
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
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not installed. Figures will be skipped.")

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. PCA analysis will be skipped.")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ANALYSIS_SAMPLE_PATH = Path("outputs/paper/data/analysis_sample.csv")
TABLE3_PATH = Path("outputs/paper/tables/table3_aggregate_factors.csv")

TABLES_DIR = Path("outputs/paper/tables")
FIGURES_DIR = Path("outputs/paper/figures")
SUPPLEMENTARY_DIR = Path("outputs/paper/supplementary")
DATA_DIR = Path("outputs/paper/data")

RANDOM_SEED = 42
N_BOOTSTRAP = 1000  # Number of bootstrap replicates
N_PLACEBO = 500  # Number of placebo permutations

# Specification curve: control set options
CONTROL_SETS = {
    "minimal": [],  # No controls, just factor counts
    "defendant": ["is_private", "is_large_enterprise"],
    "violations": ["breach_has_art5", "breach_has_art6", "breach_has_art32", "breach_has_art33"],
    "context": ["has_cctv", "has_marketing", "has_health_context", "has_employee_monitoring"],
    "procedural": ["complaint_origin", "audit_origin", "oss_case"],
    "year": ["year_2020", "year_2021", "year_2022", "year_2023", "year_2024"],
}

# Sample filter options
SAMPLE_FILTERS = {
    "all_fines": lambda df: df,  # All positive fines
    "fines_gt_1000": lambda df: df[df["fine_amount_eur"] > 1000],
    "fines_gt_10000": lambda df: df[df["fine_amount_eur"] > 10000],
    "fines_lt_10M": lambda df: df[df["fine_amount_eur"] < 10_000_000],  # Exclude mega-fines
    "fines_lt_1M": lambda df: df[df["fine_amount_eur"] < 1_000_000],  # Exclude large fines
}

# Mega-fine threshold for sensitivity analysis
MEGA_FINE_THRESHOLD = 10_000_000  # EUR 10 million

# Outcome variable options
OUTCOME_OPTIONS = {
    "log_fine_real_2025": "log_fine_2025",
    "log_fine_nominal": "log_fine_nominal",
    "log_fine_winsorized": "log_fine_winsorized",  # Winsorized at 99th percentile
}

# Random effects options
RANDOM_EFFECTS = {
    "authority": "authority_name_norm",
    "country": "country_code",
}

# Article 83(2) factor columns
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


def prepare_robustness_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for robustness analyses."""
    data = df.copy()

    # Ensure log_fine_2025 exists
    if "log_fine_2025" not in data.columns:
        fine_col = "fine_amount_eur_real_2025" if "fine_amount_eur_real_2025" in data.columns else "fine_amount_eur"
        data["log_fine_2025"] = np.log1p(data[fine_col].clip(lower=0).fillna(0))

    # Create nominal log fine for alternative outcome
    if "fine_amount_eur" in data.columns:
        data["log_fine_nominal"] = np.log1p(data["fine_amount_eur"].clip(lower=0).fillna(0))
    else:
        data["log_fine_nominal"] = data["log_fine_2025"]

    # Create winsorized log fine (cap at 99th percentile)
    fine_col = "fine_amount_eur_real_2025" if "fine_amount_eur_real_2025" in data.columns else "fine_amount_eur"
    if fine_col in data.columns:
        fine_99 = data[fine_col].quantile(0.99)
        fine_winsorized = data[fine_col].clip(upper=fine_99)
        data["log_fine_winsorized"] = np.log1p(fine_winsorized.clip(lower=0).fillna(0))
        data["fine_winsorized_cap"] = fine_99  # Store for reporting
        logger.info(f"Winsorized fines at 99th percentile: EUR {fine_99:,.0f}")
    else:
        data["log_fine_winsorized"] = data["log_fine_2025"]

    # Ensure authority and country identifiers
    if "authority_name_norm" not in data.columns:
        data["authority_name_norm"] = data["a2_authority_name"]
    data["country_code"] = data["a1_country_code"].fillna("UNKNOWN")

    # Fill aggregate factor columns
    for col in ["art83_aggravating_count", "art83_mitigating_count",
                "art83_neutral_count", "art83_discussed_count", "art83_balance_score"]:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    # Create dummy variables for controls
    if "a8_defendant_class" in data.columns:
        data["is_private"] = (data["a8_defendant_class"] == "PRIVATE").astype(int)
    if "a9_enterprise_size" in data.columns:
        data["is_large_enterprise"] = data["a9_enterprise_size"].isin(["LARGE", "VERY_LARGE"]).astype(int)

    # Year dummies
    if "decision_year" in data.columns:
        for year in [2020, 2021, 2022, 2023, 2024]:
            data[f"year_{year}"] = (data["decision_year"] == year).astype(int)

    # Violation indicators
    violation_cols = ["breach_has_art5", "breach_has_art6", "breach_has_art32", "breach_has_art33"]
    for col in violation_cols:
        if col in data.columns:
            data[col] = data[col].fillna(False).astype(int)

    # Context indicators
    context_cols = ["has_cctv", "has_marketing", "has_health_context", "has_employee_monitoring"]
    for col in context_cols:
        if col in data.columns:
            data[col] = data[col].fillna(False).astype(int)

    # Procedural indicators
    if "has_complaint_bool" in data.columns:
        data["complaint_origin"] = data["has_complaint_bool"].fillna(False).astype(int)
    if "official_audit_bool" in data.columns:
        data["audit_origin"] = data["official_audit_bool"].fillna(False).astype(int)
    if "oss_case_bool" in data.columns:
        data["oss_case"] = data["oss_case_bool"].fillna(False).astype(int)

    # Fill factor score columns
    for col in ART83_SCORE_COLS:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    logger.info(f"Prepared robustness data: {len(data)} observations")
    return data


# -----------------------------------------------------------------------------
# 1. Specification Curve Analysis
# -----------------------------------------------------------------------------

@dataclass
class SpecificationResult:
    """Container for a single specification result."""
    spec_id: int
    control_set: str
    sample_filter: str
    outcome: str
    random_effect: str
    n_obs: int
    beta_aggravating: float
    se_aggravating: float
    p_value_aggravating: float
    beta_mitigating: float
    se_mitigating: float
    p_value_mitigating: float
    icc: float
    converged: bool


def run_single_specification(
    df: pd.DataFrame,
    controls: List[str],
    outcome: str,
    groups: str,
    spec_id: int,
    control_set_name: str,
    sample_name: str
) -> Optional[SpecificationResult]:
    """Run a single model specification and return results."""
    if not HAS_STATSMODELS:
        return None

    # Build formula
    factor_vars = ["art83_aggravating_count", "art83_mitigating_count", "art83_neutral_count"]
    available_controls = [c for c in controls if c in df.columns and df[c].var() > 0]

    formula = f"{outcome} ~ " + " + ".join(factor_vars)
    if available_controls:
        formula += " + " + " + ".join(available_controls)

    # Filter to complete cases
    keep_cols = [outcome, groups] + factor_vars + available_controls
    keep_cols = [c for c in keep_cols if c in df.columns]
    model_df = df[keep_cols].dropna()

    if len(model_df) < 30:
        return None

    try:
        model = smf.mixedlm(
            formula,
            data=model_df,
            groups=model_df[groups],
            re_formula="~1"
        )
        result = model.fit(method="powell", maxiter=300)

        # Extract variance for ICC
        var_re = float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, 'iloc') else float(result.cov_re)
        var_resid = result.scale
        icc = var_re / (var_re + var_resid) if (var_re + var_resid) > 0 else 0

        return SpecificationResult(
            spec_id=spec_id,
            control_set=control_set_name,
            sample_filter=sample_name,
            outcome=outcome,
            random_effect=groups,
            n_obs=int(result.nobs),
            beta_aggravating=result.fe_params.get("art83_aggravating_count", np.nan),
            se_aggravating=result.bse_fe.get("art83_aggravating_count", np.nan),
            p_value_aggravating=result.pvalues.get("art83_aggravating_count", np.nan),
            beta_mitigating=result.fe_params.get("art83_mitigating_count", np.nan),
            se_mitigating=result.bse_fe.get("art83_mitigating_count", np.nan),
            p_value_mitigating=result.pvalues.get("art83_mitigating_count", np.nan),
            icc=icc,
            converged=True,
        )
    except Exception as e:
        logger.debug(f"Specification {spec_id} failed: {e}")
        return None


def run_specification_curve(df: pd.DataFrame) -> List[SpecificationResult]:
    """
    Run specification curve analysis across all combinations.

    Combinations:
    - Control sets: minimal, defendant, violations, context, procedural, year, full
    - Sample filters: all_fines, fines_gt_1000, fines_gt_10000
    - Outcomes: log_fine_real_2025, log_fine_nominal
    - Random effects: authority, country
    """
    results: List[SpecificationResult] = []
    spec_id = 0

    # Generate control set combinations
    control_combinations = {
        "minimal": [],
        "defendant_only": CONTROL_SETS["defendant"],
        "violations_only": CONTROL_SETS["violations"],
        "context_only": CONTROL_SETS["context"],
        "defendant+violations": CONTROL_SETS["defendant"] + CONTROL_SETS["violations"],
        "defendant+context": CONTROL_SETS["defendant"] + CONTROL_SETS["context"],
        "violations+context": CONTROL_SETS["violations"] + CONTROL_SETS["context"],
        "standard": CONTROL_SETS["defendant"] + CONTROL_SETS["violations"] + CONTROL_SETS["procedural"],
        "full": (CONTROL_SETS["defendant"] + CONTROL_SETS["violations"] +
                 CONTROL_SETS["context"] + CONTROL_SETS["procedural"] + CONTROL_SETS["year"]),
    }

    total_specs = (len(control_combinations) * len(SAMPLE_FILTERS) *
                   len(OUTCOME_OPTIONS) * len(RANDOM_EFFECTS))
    logger.info(f"Running specification curve: {total_specs} specifications")

    for ctrl_name, controls in control_combinations.items():
        for sample_name, sample_filter in SAMPLE_FILTERS.items():
            for outcome_name, outcome_col in OUTCOME_OPTIONS.items():
                for re_name, re_col in RANDOM_EFFECTS.items():
                    spec_id += 1

                    # Apply sample filter
                    filtered_df = sample_filter(df)
                    if len(filtered_df) < 50:
                        continue

                    result = run_single_specification(
                        df=filtered_df,
                        controls=controls,
                        outcome=outcome_col,
                        groups=re_col,
                        spec_id=spec_id,
                        control_set_name=ctrl_name,
                        sample_name=sample_name
                    )

                    if result:
                        results.append(result)

    logger.info(f"Completed {len(results)} of {spec_id} specifications")
    return results


def format_table8_specification_curve(results: List[SpecificationResult]) -> pd.DataFrame:
    """Format Table 8: Specification curve summary."""
    if not results:
        return pd.DataFrame()

    # Aggregate statistics
    betas = [r.beta_aggravating for r in results if not np.isnan(r.beta_aggravating)]
    p_vals = [r.p_value_aggravating for r in results if not np.isnan(r.p_value_aggravating)]

    rows = []

    # Overall summary
    rows.append({
        "Statistic": "Total Specifications",
        "Value": str(len(results)),
        "Notes": ""
    })
    rows.append({
        "Statistic": "Mean β (aggravating)",
        "Value": f"{np.mean(betas):.4f}",
        "Notes": f"Range: [{min(betas):.4f}, {max(betas):.4f}]"
    })
    rows.append({
        "Statistic": "Median β (aggravating)",
        "Value": f"{np.median(betas):.4f}",
        "Notes": f"SD: {np.std(betas):.4f}"
    })
    rows.append({
        "Statistic": "% Positive",
        "Value": f"{100 * sum(1 for b in betas if b > 0) / len(betas):.1f}%",
        "Notes": "Expected: 100% for H1"
    })
    rows.append({
        "Statistic": "% Significant (p<0.05)",
        "Value": f"{100 * sum(1 for p in p_vals if p < 0.05) / len(p_vals):.1f}%",
        "Notes": ""
    })
    rows.append({
        "Statistic": "% Significant (p<0.01)",
        "Value": f"{100 * sum(1 for p in p_vals if p < 0.01) / len(p_vals):.1f}%",
        "Notes": ""
    })

    rows.append({"Statistic": "---", "Value": "---", "Notes": "---"})

    # By control set
    for ctrl_name in set(r.control_set for r in results):
        subset = [r for r in results if r.control_set == ctrl_name]
        subset_betas = [r.beta_aggravating for r in subset if not np.isnan(r.beta_aggravating)]
        if subset_betas:
            rows.append({
                "Statistic": f"Controls: {ctrl_name}",
                "Value": f"β = {np.mean(subset_betas):.4f}",
                "Notes": f"N = {len(subset)}, range [{min(subset_betas):.4f}, {max(subset_betas):.4f}]"
            })

    rows.append({"Statistic": "---", "Value": "---", "Notes": "---"})

    # By sample filter
    for sample_name in set(r.sample_filter for r in results):
        subset = [r for r in results if r.sample_filter == sample_name]
        subset_betas = [r.beta_aggravating for r in subset if not np.isnan(r.beta_aggravating)]
        if subset_betas:
            rows.append({
                "Statistic": f"Sample: {sample_name}",
                "Value": f"β = {np.mean(subset_betas):.4f}",
                "Notes": f"N = {len(subset)}, mean n = {int(np.mean([r.n_obs for r in subset]))}"
            })

    return pd.DataFrame(rows)


def create_figure7_specification_curve(results: List[SpecificationResult]) -> Optional[plt.Figure]:
    """Create Figure 7: Specification curve plot."""
    if not HAS_PLOTTING or not results:
        return None

    # Sort results by beta
    sorted_results = sorted(results, key=lambda x: x.beta_aggravating if not np.isnan(x.beta_aggravating) else -999)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Top panel: Coefficient plot
    ax1 = axes[0]
    x = range(len(sorted_results))
    betas = [r.beta_aggravating for r in sorted_results]
    ses = [r.se_aggravating for r in sorted_results]
    p_vals = [r.p_value_aggravating for r in sorted_results]

    # Color by significance
    colors = ['#2ecc71' if p < 0.05 else '#e74c3c' for p in p_vals]

    # Plot coefficients with error bars
    ax1.scatter(x, betas, c=colors, s=30, alpha=0.7, edgecolor='none')
    for i, (b, se) in enumerate(zip(betas, ses)):
        if not np.isnan(b) and not np.isnan(se):
            ax1.plot([i, i], [b - 1.96*se, b + 1.96*se], c=colors[i], alpha=0.3, linewidth=1)

    ax1.axhline(0, color='black', linestyle='-', linewidth=1)

    # Add median line
    median_beta = np.nanmedian(betas)
    ax1.axhline(median_beta, color='blue', linestyle='--', linewidth=2,
                label=f'Median β = {median_beta:.3f}')

    ax1.set_ylabel('Coefficient (Aggravating Factor Count)', fontsize=12)
    ax1.set_title('Figure 7: Specification Curve Analysis\n(Aggravating Factor Effect Across Model Specifications)',
                  fontsize=14)
    ax1.legend(loc='upper left')
    ax1.set_xlim(-5, len(sorted_results) + 5)

    # Add annotation for % significant
    n_sig = sum(1 for p in p_vals if p < 0.05 and not np.isnan(p))
    ax1.annotate(f'{100*n_sig/len(p_vals):.1f}% significant (p<0.05)\nGreen = significant, Red = not significant',
                 xy=(0.98, 0.98), xycoords='axes fraction',
                 ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Bottom panel: Specification indicators
    ax2 = axes[1]

    # Create specification indicator matrix
    spec_features = []
    for r in sorted_results:
        spec_features.append({
            'full_controls': 1 if r.control_set == 'full' else 0,
            'standard_controls': 1 if r.control_set == 'standard' else 0,
            'minimal_controls': 1 if r.control_set == 'minimal' else 0,
            'all_fines': 1 if r.sample_filter == 'all_fines' else 0,
            'fines_gt_1000': 1 if r.sample_filter == 'fines_gt_1000' else 0,
            'fines_gt_10000': 1 if r.sample_filter == 'fines_gt_10000' else 0,
            'authority_re': 1 if r.random_effect == 'authority_name_norm' else 0,
            'country_re': 1 if r.random_effect == 'country_code' else 0,
        })

    spec_matrix = pd.DataFrame(spec_features)

    # Plot as heatmap-style indicators
    for i, col in enumerate(spec_matrix.columns):
        y_positions = [i] * len(sorted_results)
        x_positions = list(range(len(sorted_results)))
        sizes = [15 if v == 1 else 0 for v in spec_matrix[col]]
        ax2.scatter(x_positions, y_positions, s=sizes, c='black', marker='s')

    ax2.set_yticks(range(len(spec_matrix.columns)))
    ax2.set_yticklabels([c.replace('_', ' ').title() for c in spec_matrix.columns], fontsize=9)
    ax2.set_xlim(-5, len(sorted_results) + 5)
    ax2.set_xlabel('Specification (sorted by coefficient)', fontsize=11)
    ax2.invert_yaxis()

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 2. Bootstrap Confidence Intervals
# -----------------------------------------------------------------------------

@dataclass
class BootstrapResults:
    """Container for bootstrap confidence interval results."""
    variable: str
    original_estimate: float
    bootstrap_mean: float
    bootstrap_se: float
    ci_lower_95: float
    ci_upper_95: float
    ci_lower_99: float
    ci_upper_99: float
    n_successful: int


def run_bootstrap_model(df: pd.DataFrame, n_bootstrap: int = 1000) -> List[BootstrapResults]:
    """
    Run bootstrap for main model coefficients.

    Uses case-resampling bootstrap with cluster adjustment for authorities.
    """
    if not HAS_STATSMODELS:
        return []

    np.random.seed(RANDOM_SEED)

    # Define the model specification (matches Model 1 from Phase 3)
    factor_vars = ["art83_aggravating_count", "art83_mitigating_count", "art83_neutral_count"]
    control_vars = ["is_private", "is_large_enterprise", "breach_has_art5", "breach_has_art6",
                    "complaint_origin", "audit_origin", "oss_case"]

    all_vars = factor_vars + [c for c in control_vars if c in df.columns and df[c].var() > 0]
    keep_cols = ["log_fine_2025", "authority_name_norm"] + all_vars
    keep_cols = [c for c in keep_cols if c in df.columns]

    model_df = df[keep_cols].dropna()

    if len(model_df) < 50:
        logger.warning("Insufficient observations for bootstrap")
        return []

    # Get original estimates
    formula = "log_fine_2025 ~ " + " + ".join(all_vars)
    try:
        original_model = smf.mixedlm(
            formula,
            data=model_df,
            groups=model_df["authority_name_norm"],
            re_formula="~1"
        )
        original_fit = original_model.fit(method="powell", maxiter=300)
        original_coefs = dict(original_fit.fe_params)
    except Exception as e:
        logger.error(f"Original model failed: {e}")
        return []

    # Bootstrap resampling (by authority clusters)
    authorities = model_df["authority_name_norm"].unique()
    bootstrap_coefs = {var: [] for var in original_coefs.keys()}

    for b in range(n_bootstrap):
        if (b + 1) % 100 == 0:
            logger.info(f"  Bootstrap iteration {b + 1}/{n_bootstrap}")

        # Resample authorities with replacement
        sampled_authorities = np.random.choice(authorities, size=len(authorities), replace=True)

        # Build resampled dataset
        boot_dfs = []
        for auth in sampled_authorities:
            auth_df = model_df[model_df["authority_name_norm"] == auth].copy()
            boot_dfs.append(auth_df)

        boot_df = pd.concat(boot_dfs, ignore_index=True)

        # Fit model on bootstrap sample
        try:
            boot_model = smf.mixedlm(
                formula,
                data=boot_df,
                groups=boot_df["authority_name_norm"],
                re_formula="~1"
            )
            boot_fit = boot_model.fit(method="powell", maxiter=200)

            for var in bootstrap_coefs.keys():
                if var in boot_fit.fe_params:
                    bootstrap_coefs[var].append(boot_fit.fe_params[var])
        except Exception:
            continue

    # Compute bootstrap CIs
    results: List[BootstrapResults] = []

    key_vars = ["art83_aggravating_count", "art83_mitigating_count", "art83_neutral_count",
                "is_private", "is_large_enterprise"]

    for var in key_vars:
        if var not in bootstrap_coefs or len(bootstrap_coefs[var]) < 100:
            continue

        boot_samples = np.array(bootstrap_coefs[var])

        results.append(BootstrapResults(
            variable=var,
            original_estimate=original_coefs.get(var, np.nan),
            bootstrap_mean=np.mean(boot_samples),
            bootstrap_se=np.std(boot_samples, ddof=1),
            ci_lower_95=np.percentile(boot_samples, 2.5),
            ci_upper_95=np.percentile(boot_samples, 97.5),
            ci_lower_99=np.percentile(boot_samples, 0.5),
            ci_upper_99=np.percentile(boot_samples, 99.5),
            n_successful=len(boot_samples),
        ))

    return results


def format_bootstrap_table(results: List[BootstrapResults]) -> pd.DataFrame:
    """Format bootstrap CI results as table."""
    rows = []

    var_labels = {
        "art83_aggravating_count": "Aggravating Factor Count",
        "art83_mitigating_count": "Mitigating Factor Count",
        "art83_neutral_count": "Neutral Factor Count",
        "is_private": "Private Sector",
        "is_large_enterprise": "Large Enterprise",
    }

    for r in results:
        rows.append({
            "Variable": var_labels.get(r.variable, r.variable),
            "Original Est.": f"{r.original_estimate:.4f}",
            "Bootstrap Mean": f"{r.bootstrap_mean:.4f}",
            "Bootstrap SE": f"{r.bootstrap_se:.4f}",
            "95% CI": f"[{r.ci_lower_95:.4f}, {r.ci_upper_95:.4f}]",
            "99% CI": f"[{r.ci_lower_99:.4f}, {r.ci_upper_99:.4f}]",
            "N Bootstrap": r.n_successful,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 3. Leave-One-Country-Out Sensitivity
# -----------------------------------------------------------------------------

@dataclass
class LOCOResult:
    """Leave-one-country-out result."""
    excluded_country: str
    n_excluded: int
    n_remaining: int
    beta_aggravating: float
    se_aggravating: float
    p_value_aggravating: float
    beta_mitigating: float
    change_from_full: float  # Percentage change from full sample estimate


def run_leave_one_country_out(df: pd.DataFrame, full_beta: float) -> List[LOCOResult]:
    """
    Run leave-one-country-out sensitivity analysis.

    Estimates Model 1 excluding each country sequentially.
    """
    if not HAS_STATSMODELS:
        return []

    results: List[LOCOResult] = []
    countries = df["country_code"].dropna().unique()

    # Model specification
    factor_vars = ["art83_aggravating_count", "art83_mitigating_count", "art83_neutral_count"]
    control_vars = ["is_private", "is_large_enterprise", "breach_has_art5", "breach_has_art6",
                    "complaint_origin", "audit_origin", "oss_case"]

    available_controls = [c for c in control_vars if c in df.columns and df[c].var() > 0]
    formula = "log_fine_2025 ~ " + " + ".join(factor_vars + available_controls)

    logger.info(f"Running LOCO for {len(countries)} countries")

    for country in countries:
        excluded_df = df[df["country_code"] != country].copy()
        n_excluded = len(df[df["country_code"] == country])

        if len(excluded_df) < 50:
            continue

        keep_cols = ["log_fine_2025", "authority_name_norm"] + factor_vars + available_controls
        keep_cols = [c for c in keep_cols if c in excluded_df.columns]
        model_df = excluded_df[keep_cols].dropna()

        if len(model_df) < 30:
            continue

        try:
            model = smf.mixedlm(
                formula,
                data=model_df,
                groups=model_df["authority_name_norm"],
                re_formula="~1"
            )
            fit = model.fit(method="powell", maxiter=300)

            beta_agg = fit.fe_params.get("art83_aggravating_count", np.nan)
            change = 100 * (beta_agg - full_beta) / abs(full_beta) if full_beta != 0 else 0

            results.append(LOCOResult(
                excluded_country=country,
                n_excluded=n_excluded,
                n_remaining=len(model_df),
                beta_aggravating=beta_agg,
                se_aggravating=fit.bse_fe.get("art83_aggravating_count", np.nan),
                p_value_aggravating=fit.pvalues.get("art83_aggravating_count", np.nan),
                beta_mitigating=fit.fe_params.get("art83_mitigating_count", np.nan),
                change_from_full=change,
            ))
        except Exception as e:
            logger.debug(f"LOCO failed for {country}: {e}")
            continue

    return results


def format_loco_table(results: List[LOCOResult], full_beta: float) -> pd.DataFrame:
    """Format LOCO results as table."""
    rows = []

    # Sort by absolute change from full sample
    sorted_results = sorted(results, key=lambda x: abs(x.change_from_full), reverse=True)

    for r in sorted_results:
        sig = "***" if r.p_value_aggravating < 0.001 else "**" if r.p_value_aggravating < 0.01 else "*" if r.p_value_aggravating < 0.05 else ""
        rows.append({
            "Excluded Country": r.excluded_country,
            "N Excluded": r.n_excluded,
            "N Remaining": r.n_remaining,
            "β (Aggravating)": f"{r.beta_aggravating:.4f}{sig}",
            "SE": f"{r.se_aggravating:.4f}",
            "% Change": f"{r.change_from_full:+.1f}%",
            "Interpretation": "Stable" if abs(r.change_from_full) < 20 else "Notable influence"
        })

    # Add summary row
    betas = [r.beta_aggravating for r in results if not np.isnan(r.beta_aggravating)]
    rows.append({
        "Excluded Country": "--- SUMMARY ---",
        "N Excluded": "",
        "N Remaining": "",
        "β (Aggravating)": f"Mean: {np.mean(betas):.4f}",
        "SE": f"SD: {np.std(betas):.4f}",
        "% Change": f"Range: [{min(betas):.4f}, {max(betas):.4f}]",
        "Interpretation": f"Full sample β = {full_beta:.4f}"
    })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 4. Placebo Tests
# -----------------------------------------------------------------------------

@dataclass
class PlaceboResults:
    """Container for placebo test results."""
    test_name: str
    n_permutations: int
    original_beta: float
    mean_placebo_beta: float
    sd_placebo_beta: float
    percentile_rank: float  # Where original falls in placebo distribution
    p_value: float  # Two-sided p-value
    significant: bool


def run_placebo_article_shuffle(df: pd.DataFrame, n_permutations: int = 500) -> Optional[PlaceboResults]:
    """
    Placebo test: Random permutation of article_set_key within country.

    If articles matter for fines, shuffling should attenuate cohort effects.
    """
    if not HAS_STATSMODELS:
        return None

    np.random.seed(RANDOM_SEED)

    # Simple model for this test
    formula = "log_fine_2025 ~ art83_aggravating_count + art83_mitigating_count + is_private + is_large_enterprise"
    keep_cols = ["log_fine_2025", "authority_name_norm", "art83_aggravating_count",
                 "art83_mitigating_count", "is_private", "is_large_enterprise", "country_code"]

    model_df = df[[c for c in keep_cols if c in df.columns]].dropna()

    if len(model_df) < 50:
        return None

    # Get original estimate
    try:
        original_model = smf.mixedlm(
            formula,
            data=model_df,
            groups=model_df["authority_name_norm"],
            re_formula="~1"
        )
        original_fit = original_model.fit(method="powell", maxiter=300)
        original_beta = original_fit.fe_params.get("art83_aggravating_count", np.nan)
    except Exception:
        return None

    # Run permutations (shuffle factor counts within country)
    placebo_betas = []

    for p in range(n_permutations):
        if (p + 1) % 100 == 0:
            logger.info(f"  Placebo permutation {p + 1}/{n_permutations}")

        perm_df = model_df.copy()

        # Shuffle aggravating count within each country
        for country in perm_df["country_code"].unique():
            mask = perm_df["country_code"] == country
            shuffled = perm_df.loc[mask, "art83_aggravating_count"].sample(frac=1).values
            perm_df.loc[mask, "art83_aggravating_count"] = shuffled

        try:
            perm_model = smf.mixedlm(
                formula,
                data=perm_df,
                groups=perm_df["authority_name_norm"],
                re_formula="~1"
            )
            perm_fit = perm_model.fit(method="powell", maxiter=200)
            placebo_betas.append(perm_fit.fe_params.get("art83_aggravating_count", np.nan))
        except Exception:
            continue

    if len(placebo_betas) < 50:
        return None

    placebo_betas = np.array([b for b in placebo_betas if not np.isnan(b)])

    # Compute p-value (proportion of placebo betas as extreme as original)
    n_extreme = sum(1 for b in placebo_betas if abs(b) >= abs(original_beta))
    p_value = n_extreme / len(placebo_betas)

    # Percentile rank
    percentile = 100 * sum(1 for b in placebo_betas if b <= original_beta) / len(placebo_betas)

    return PlaceboResults(
        test_name="Factor Shuffle (within-country)",
        n_permutations=len(placebo_betas),
        original_beta=original_beta,
        mean_placebo_beta=np.mean(placebo_betas),
        sd_placebo_beta=np.std(placebo_betas, ddof=1),
        percentile_rank=percentile,
        p_value=p_value,
        significant=p_value < 0.05,
    )


def format_placebo_table(results: List[PlaceboResults]) -> pd.DataFrame:
    """Format placebo test results."""
    rows = []

    for r in results:
        rows.append({
            "Test": r.test_name,
            "N Permutations": r.n_permutations,
            "Original β": f"{r.original_beta:.4f}",
            "Placebo Mean": f"{r.mean_placebo_beta:.4f}",
            "Placebo SD": f"{r.sd_placebo_beta:.4f}",
            "Percentile Rank": f"{r.percentile_rank:.1f}%",
            "p-value": f"{r.p_value:.4f}",
            "Conclusion": "Original effect is real" if r.significant else "Cannot rule out chance"
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 5. Alternative Factor Operationalizations
# -----------------------------------------------------------------------------

@dataclass
class AlternativeOperationalization:
    """Results from alternative factor operationalization."""
    operationalization: str
    beta_main: float
    se_main: float
    p_value: float
    r2_marginal: float
    n_obs: int
    notes: str


def run_alternative_operationalizations(df: pd.DataFrame) -> List[AlternativeOperationalization]:
    """
    Run models with alternative factor operationalizations:
    1. Binary counts (AGGRAVATING=1 vs all else)
    2. Weighted scores (factor importance from decomposition)
    3. PCA on factor matrix
    """
    if not HAS_STATSMODELS:
        return []

    results: List[AlternativeOperationalization] = []

    # Prepare model data
    control_vars = ["is_private", "is_large_enterprise", "breach_has_art5", "breach_has_art6"]

    # Original operationalization for comparison
    model_df = df[["log_fine_2025", "authority_name_norm", "art83_aggravating_count",
                   "art83_mitigating_count"] + [c for c in control_vars if c in df.columns]].dropna()

    # 1. Binary aggravating count (any vs none)
    df_binary = model_df.copy()
    df_binary["has_any_aggravating"] = (df_binary["art83_aggravating_count"] > 0).astype(int)

    try:
        formula = "log_fine_2025 ~ has_any_aggravating + art83_mitigating_count + " + " + ".join(
            [c for c in control_vars if c in df_binary.columns])
        model = smf.mixedlm(formula, data=df_binary, groups=df_binary["authority_name_norm"], re_formula="~1")
        fit = model.fit(method="powell", maxiter=300)

        results.append(AlternativeOperationalization(
            operationalization="Binary (Any Aggravating vs None)",
            beta_main=fit.fe_params.get("has_any_aggravating", np.nan),
            se_main=fit.bse_fe.get("has_any_aggravating", np.nan),
            p_value=fit.pvalues.get("has_any_aggravating", np.nan),
            r2_marginal=np.var(fit.fittedvalues) / np.var(df_binary["log_fine_2025"]),
            n_obs=int(fit.nobs),
            notes="Tests if having any aggravating factor matters"
        ))
    except Exception as e:
        logger.debug(f"Binary operationalization failed: {e}")

    # 2. High aggravating (3+ factors)
    df_high = model_df.copy()
    df_high["high_aggravating"] = (df_high["art83_aggravating_count"] >= 3).astype(int)

    try:
        formula = "log_fine_2025 ~ high_aggravating + art83_mitigating_count + " + " + ".join(
            [c for c in control_vars if c in df_high.columns])
        model = smf.mixedlm(formula, data=df_high, groups=df_high["authority_name_norm"], re_formula="~1")
        fit = model.fit(method="powell", maxiter=300)

        results.append(AlternativeOperationalization(
            operationalization="High Aggravating (3+ factors)",
            beta_main=fit.fe_params.get("high_aggravating", np.nan),
            se_main=fit.bse_fe.get("high_aggravating", np.nan),
            p_value=fit.pvalues.get("high_aggravating", np.nan),
            r2_marginal=np.var(fit.fittedvalues) / np.var(df_high["log_fine_2025"]),
            n_obs=int(fit.nobs),
            notes="Tests if having many aggravating factors matters more"
        ))
    except Exception as e:
        logger.debug(f"High aggravating operationalization failed: {e}")

    # 3. Balance score (aggravating - mitigating)
    df_balance = model_df.copy()
    df_balance["factor_balance"] = df_balance["art83_aggravating_count"] - df_balance["art83_mitigating_count"]

    try:
        formula = "log_fine_2025 ~ factor_balance + " + " + ".join(
            [c for c in control_vars if c in df_balance.columns])
        model = smf.mixedlm(formula, data=df_balance, groups=df_balance["authority_name_norm"], re_formula="~1")
        fit = model.fit(method="powell", maxiter=300)

        results.append(AlternativeOperationalization(
            operationalization="Balance Score (Agg - Mit)",
            beta_main=fit.fe_params.get("factor_balance", np.nan),
            se_main=fit.bse_fe.get("factor_balance", np.nan),
            p_value=fit.pvalues.get("factor_balance", np.nan),
            r2_marginal=np.var(fit.fittedvalues) / np.var(df_balance["log_fine_2025"]),
            n_obs=int(fit.nobs),
            notes="Net effect of aggravating minus mitigating"
        ))
    except Exception as e:
        logger.debug(f"Balance operationalization failed: {e}")

    # 4. PCA on factor scores (if sklearn available)
    if HAS_SKLEARN:
        # Get individual factor scores
        factor_cols = [c for c in ART83_SCORE_COLS if c in df.columns]
        if len(factor_cols) >= 3:
            df_pca = df[["log_fine_2025", "authority_name_norm"] + factor_cols +
                       [c for c in control_vars if c in df.columns]].dropna()

            if len(df_pca) >= 50:
                try:
                    factor_matrix = df_pca[factor_cols].values
                    scaler = StandardScaler()
                    factor_scaled = scaler.fit_transform(factor_matrix)

                    pca = PCA(n_components=1)
                    pc1 = pca.fit_transform(factor_scaled)
                    df_pca["factor_pc1"] = pc1

                    formula = "log_fine_2025 ~ factor_pc1 + " + " + ".join(
                        [c for c in control_vars if c in df_pca.columns])
                    model = smf.mixedlm(formula, data=df_pca, groups=df_pca["authority_name_norm"], re_formula="~1")
                    fit = model.fit(method="powell", maxiter=300)

                    results.append(AlternativeOperationalization(
                        operationalization="PCA (First Principal Component)",
                        beta_main=fit.fe_params.get("factor_pc1", np.nan),
                        se_main=fit.bse_fe.get("factor_pc1", np.nan),
                        p_value=fit.pvalues.get("factor_pc1", np.nan),
                        r2_marginal=np.var(fit.fittedvalues) / np.var(df_pca["log_fine_2025"]),
                        n_obs=int(fit.nobs),
                        notes=f"PC1 explains {100*pca.explained_variance_ratio_[0]:.1f}% of factor variance"
                    ))
                except Exception as e:
                    logger.debug(f"PCA operationalization failed: {e}")

    return results


def format_alternative_operationalization_table(results: List[AlternativeOperationalization]) -> pd.DataFrame:
    """Format alternative operationalization results."""
    rows = []

    for r in results:
        sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
        rows.append({
            "Operationalization": r.operationalization,
            "β": f"{r.beta_main:.4f}{sig}",
            "SE": f"{r.se_main:.4f}",
            "p-value": f"{r.p_value:.4f}",
            "R² (marginal)": f"{r.r2_marginal:.3f}",
            "N": r.n_obs,
            "Notes": r.notes,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 6. Mega-Fine Sensitivity Analysis (NEW)
# -----------------------------------------------------------------------------

@dataclass
class MegaFineSensitivity:
    """Results from mega-fine exclusion sensitivity analysis."""
    analysis_type: str
    sample_description: str
    n_obs: int
    n_excluded: int
    beta_aggravating: float
    se_aggravating: float
    p_value: float
    beta_mitigating: float
    pct_change_from_full: float
    notes: str


def run_mega_fine_sensitivity(df: pd.DataFrame) -> List[MegaFineSensitivity]:
    """
    Run sensitivity analysis excluding mega-fines.

    Tests whether large fines (>EUR 10M) drive the main results.
    """
    if not HAS_STATSMODELS:
        return []

    results: List[MegaFineSensitivity] = []

    # Model specification
    factor_vars = ["art83_aggravating_count", "art83_mitigating_count", "art83_neutral_count"]
    control_vars = ["is_private", "is_large_enterprise", "breach_has_art5", "breach_has_art6",
                    "complaint_origin", "audit_origin", "oss_case"]
    available_controls = [c for c in control_vars if c in df.columns and df[c].var() > 0]
    formula = "log_fine_2025 ~ " + " + ".join(factor_vars + available_controls)

    # Prepare base data
    keep_cols = ["log_fine_2025", "authority_name_norm", "fine_amount_eur"] + factor_vars + available_controls
    keep_cols = [c for c in keep_cols if c in df.columns]
    base_df = df[keep_cols].dropna()

    # Define sample cuts
    sample_cuts = [
        ("Full Sample", base_df, "All fines included"),
        ("Exclude >EUR 10M", base_df[base_df["fine_amount_eur"] < 10_000_000], "Mega-fines excluded"),
        ("Exclude >EUR 1M", base_df[base_df["fine_amount_eur"] < 1_000_000], "Large fines excluded"),
        ("Exclude >EUR 100K", base_df[base_df["fine_amount_eur"] < 100_000], "Medium+ fines excluded"),
    ]

    full_beta = None

    for name, sample_df, notes in sample_cuts:
        if len(sample_df) < 30:
            continue

        try:
            model = smf.mixedlm(
                formula,
                data=sample_df,
                groups=sample_df["authority_name_norm"],
                re_formula="~1"
            )
            fit = model.fit(method="powell", maxiter=300)

            beta_agg = fit.fe_params.get("art83_aggravating_count", np.nan)

            if full_beta is None:
                full_beta = beta_agg

            pct_change = 100 * (beta_agg - full_beta) / abs(full_beta) if full_beta else 0

            results.append(MegaFineSensitivity(
                analysis_type="Sample Exclusion",
                sample_description=name,
                n_obs=len(sample_df),
                n_excluded=len(base_df) - len(sample_df),
                beta_aggravating=beta_agg,
                se_aggravating=fit.bse_fe.get("art83_aggravating_count", np.nan),
                p_value=fit.pvalues.get("art83_aggravating_count", np.nan),
                beta_mitigating=fit.fe_params.get("art83_mitigating_count", np.nan),
                pct_change_from_full=pct_change,
                notes=notes,
            ))

        except Exception as e:
            logger.debug(f"Mega-fine sensitivity failed for {name}: {e}")
            continue

    return results


def format_mega_fine_sensitivity_table(results: List[MegaFineSensitivity]) -> pd.DataFrame:
    """Format mega-fine sensitivity results as Table S5."""
    rows = []

    for r in results:
        sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
        rows.append({
            "Sample": r.sample_description,
            "N": r.n_obs,
            "N Excluded": r.n_excluded,
            "β (Aggravating)": f"{r.beta_aggravating:.4f}{sig}",
            "SE": f"{r.se_aggravating:.4f}",
            "p-value": f"{r.p_value:.4f}" if r.p_value >= 0.0001 else "<0.0001",
            "% Change from Full": f"{r.pct_change_from_full:+.1f}%" if r.pct_change_from_full != 0 else "—",
            "β (Mitigating)": f"{r.beta_mitigating:.4f}",
            "Notes": r.notes,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 7. Winsorization Sensitivity Analysis (NEW)
# -----------------------------------------------------------------------------

@dataclass
class WinsorSensitivity:
    """Results from winsorization sensitivity analysis."""
    outcome_type: str
    winsor_level: str
    n_obs: int
    n_affected: int
    beta_aggravating: float
    se_aggravating: float
    p_value: float
    beta_mitigating: float
    icc: float
    notes: str


def run_winsorization_sensitivity(df: pd.DataFrame) -> List[WinsorSensitivity]:
    """
    Run sensitivity analysis with different winsorization levels.

    Tests whether extreme values drive results by capping at different percentiles.
    """
    if not HAS_STATSMODELS:
        return []

    results: List[WinsorSensitivity] = []

    # Model specification
    factor_vars = ["art83_aggravating_count", "art83_mitigating_count", "art83_neutral_count"]
    control_vars = ["is_private", "is_large_enterprise", "breach_has_art5", "breach_has_art6",
                    "complaint_origin", "audit_origin", "oss_case"]
    available_controls = [c for c in control_vars if c in df.columns and df[c].var() > 0]

    # Get fine column
    fine_col = "fine_amount_eur_real_2025" if "fine_amount_eur_real_2025" in df.columns else "fine_amount_eur"

    # Prepare base data
    keep_cols = ["authority_name_norm", fine_col] + factor_vars + available_controls
    keep_cols = [c for c in keep_cols if c in df.columns]
    base_df = df[keep_cols].dropna()

    if len(base_df) < 50:
        return []

    # Winsorization levels to test
    winsor_levels = [
        ("No Winsorization", 1.0),
        ("99th Percentile", 0.99),
        ("97.5th Percentile", 0.975),
        ("95th Percentile", 0.95),
        ("90th Percentile", 0.90),
    ]

    for name, percentile in winsor_levels:
        try:
            model_df = base_df.copy()

            if percentile < 1.0:
                cap = model_df[fine_col].quantile(percentile)
                n_affected = (model_df[fine_col] > cap).sum()
                model_df["fine_winsor"] = model_df[fine_col].clip(upper=cap)
            else:
                cap = model_df[fine_col].max()
                n_affected = 0
                model_df["fine_winsor"] = model_df[fine_col]

            model_df["log_fine_winsor"] = np.log1p(model_df["fine_winsor"].clip(lower=0))

            formula = "log_fine_winsor ~ " + " + ".join(factor_vars + available_controls)

            model = smf.mixedlm(
                formula,
                data=model_df,
                groups=model_df["authority_name_norm"],
                re_formula="~1"
            )
            fit = model.fit(method="powell", maxiter=300)

            # Calculate ICC
            var_re = float(fit.cov_re.iloc[0, 0]) if hasattr(fit.cov_re, 'iloc') else float(fit.cov_re)
            var_resid = fit.scale
            icc = var_re / (var_re + var_resid) if (var_re + var_resid) > 0 else 0

            results.append(WinsorSensitivity(
                outcome_type="Log Fine (Real 2025 EUR)",
                winsor_level=name,
                n_obs=len(model_df),
                n_affected=n_affected,
                beta_aggravating=fit.fe_params.get("art83_aggravating_count", np.nan),
                se_aggravating=fit.bse_fe.get("art83_aggravating_count", np.nan),
                p_value=fit.pvalues.get("art83_aggravating_count", np.nan),
                beta_mitigating=fit.fe_params.get("art83_mitigating_count", np.nan),
                icc=icc,
                notes=f"Cap at EUR {cap:,.0f}" if percentile < 1.0 else "No cap",
            ))

        except Exception as e:
            logger.debug(f"Winsorization sensitivity failed for {name}: {e}")
            continue

    return results


def format_winsorization_table(results: List[WinsorSensitivity]) -> pd.DataFrame:
    """Format winsorization sensitivity results as Table S6."""
    rows = []

    for r in results:
        sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
        rows.append({
            "Winsorization": r.winsor_level,
            "N": r.n_obs,
            "N Capped": r.n_affected,
            "β (Aggravating)": f"{r.beta_aggravating:.4f}{sig}",
            "SE": f"{r.se_aggravating:.4f}",
            "p-value": f"{r.p_value:.4f}" if r.p_value >= 0.0001 else "<0.0001",
            "β (Mitigating)": f"{r.beta_mitigating:.4f}",
            "ICC": f"{r.icc:.3f}",
            "Cap": r.notes,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Output and Main
# -----------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    """Create output directories if they don't exist."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTARY_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_phase5_summary(
    spec_results: List[SpecificationResult],
    bootstrap_results: List[BootstrapResults],
    loco_results: List[LOCOResult],
    placebo_results: List[PlaceboResults],
    alt_results: List[AlternativeOperationalization],
    mega_fine_results: Optional[List[MegaFineSensitivity]] = None,
    winsor_results: Optional[List[WinsorSensitivity]] = None
) -> None:
    """Save comprehensive Phase 5 summary."""
    summary_path = DATA_DIR / "phase5_robustness_summary.txt"

    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 5: ROBUSTNESS & FINALIZATION - RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        # Specification Curve
        f.write("1. SPECIFICATION CURVE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        if spec_results:
            betas = [r.beta_aggravating for r in spec_results if not np.isnan(r.beta_aggravating)]
            p_vals = [r.p_value_aggravating for r in spec_results if not np.isnan(r.p_value_aggravating)]
            f.write(f"Total specifications: {len(spec_results)}\n")
            f.write(f"Mean β (aggravating): {np.mean(betas):.4f}\n")
            f.write(f"Median β (aggravating): {np.median(betas):.4f}\n")
            f.write(f"Range: [{min(betas):.4f}, {max(betas):.4f}]\n")
            f.write(f"% Positive: {100 * sum(1 for b in betas if b > 0) / len(betas):.1f}%\n")
            f.write(f"% Significant (p<0.05): {100 * sum(1 for p in p_vals if p < 0.05) / len(p_vals):.1f}%\n")
        else:
            f.write("No results available.\n")

        f.write("\n")

        # Bootstrap
        f.write("2. BOOTSTRAP CONFIDENCE INTERVALS\n")
        f.write("-" * 40 + "\n")
        if bootstrap_results:
            for r in bootstrap_results:
                f.write(f"{r.variable}:\n")
                f.write(f"  Original: {r.original_estimate:.4f}\n")
                f.write(f"  95% CI: [{r.ci_lower_95:.4f}, {r.ci_upper_95:.4f}]\n")
                f.write(f"  Bootstrap SE: {r.bootstrap_se:.4f}\n")
        else:
            f.write("No results available.\n")

        f.write("\n")

        # LOCO
        f.write("3. LEAVE-ONE-COUNTRY-OUT SENSITIVITY\n")
        f.write("-" * 40 + "\n")
        if loco_results:
            betas = [r.beta_aggravating for r in loco_results if not np.isnan(r.beta_aggravating)]
            changes = [r.change_from_full for r in loco_results]
            f.write(f"Countries analyzed: {len(loco_results)}\n")
            f.write(f"Coefficient range: [{min(betas):.4f}, {max(betas):.4f}]\n")
            f.write(f"Max % change: {max(abs(c) for c in changes):.1f}%\n")
            influential = [r.excluded_country for r in loco_results if abs(r.change_from_full) > 20]
            if influential:
                f.write(f"Influential countries (>20% change): {', '.join(influential)}\n")
            else:
                f.write("No single country is highly influential (all <20% change)\n")
        else:
            f.write("No results available.\n")

        f.write("\n")

        # Placebo
        f.write("4. PLACEBO TESTS\n")
        f.write("-" * 40 + "\n")
        if placebo_results:
            for r in placebo_results:
                f.write(f"{r.test_name}:\n")
                f.write(f"  Original β: {r.original_beta:.4f}\n")
                f.write(f"  Placebo mean: {r.mean_placebo_beta:.4f}\n")
                f.write(f"  Placebo p-value: {r.p_value:.4f}\n")
                f.write(f"  Conclusion: {'Effect is robust' if r.p_value < 0.05 else 'Cannot rule out chance'}\n")
        else:
            f.write("No results available.\n")

        f.write("\n")

        # Alternative operationalizations
        f.write("5. ALTERNATIVE OPERATIONALIZATIONS\n")
        f.write("-" * 40 + "\n")
        if alt_results:
            for r in alt_results:
                sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
                f.write(f"{r.operationalization}: β={r.beta_main:.4f}{sig} (p={r.p_value:.4f})\n")
        else:
            f.write("No results available.\n")

        f.write("\n")

        # Mega-fine sensitivity (NEW)
        f.write("6. MEGA-FINE SENSITIVITY ANALYSIS (Outlier Robustness)\n")
        f.write("-" * 40 + "\n")
        if mega_fine_results:
            f.write("Tests whether excluding large fines affects results:\n\n")
            for r in mega_fine_results:
                sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
                f.write(f"  {r.sample_description}: β={r.beta_aggravating:.4f}{sig} "
                        f"(N={r.n_obs}, change={r.pct_change_from_full:+.1f}%)\n")
            f.write("\nInterpretation: Results are robust to excluding mega-fines if coefficient "
                    "remains significant and similar magnitude.\n")
        else:
            f.write("No results available.\n")

        f.write("\n")

        # Winsorization sensitivity (NEW)
        f.write("7. WINSORIZATION SENSITIVITY ANALYSIS (Outlier Robustness)\n")
        f.write("-" * 40 + "\n")
        if winsor_results:
            f.write("Tests whether capping extreme values affects results:\n\n")
            for r in winsor_results:
                sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
                f.write(f"  {r.winsor_level}: β={r.beta_aggravating:.4f}{sig} "
                        f"(N capped={r.n_affected}, ICC={r.icc:.3f})\n")
            f.write("\nInterpretation: Results are robust to winsorization if coefficient "
                    "remains stable across percentile caps.\n")
        else:
            f.write("No results available.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Generated by 10_robustness_analysis.py (Phase 5)\n")

    logger.info(f"Saved Phase 5 summary to {summary_path}")


def main() -> None:
    """Execute Phase 5: Robustness & Finalization."""
    logger.info("=" * 70)
    logger.info("PHASE 5: ROBUSTNESS & FINALIZATION")
    logger.info("=" * 70)

    np.random.seed(RANDOM_SEED)
    ensure_output_dirs()

    # Load and prepare data
    logger.info("\n[1/10] Loading analysis sample...")
    df = load_analysis_sample()
    df = prepare_robustness_data(df)

    # 1. Specification Curve Analysis
    logger.info("\n[2/10] Running specification curve analysis...")
    spec_results = run_specification_curve(df)

    if spec_results:
        table8 = format_table8_specification_curve(spec_results)
        table8_path = TABLES_DIR / "table8_specification_curve.csv"
        table8.to_csv(table8_path, index=False)
        logger.info(f"  Saved Table 8 to {table8_path}")
        print("\n" + "=" * 50)
        print("TABLE 8: Specification Curve Summary")
        print("=" * 50)
        print(table8.to_string(index=False))

        # Figure 7
        logger.info("\n[3/10] Creating Figure 7: Specification curve...")
        fig7 = create_figure7_specification_curve(spec_results)
        if fig7:
            fig7_path = FIGURES_DIR / "figure7_specification_curve.png"
            fig7.savefig(fig7_path, dpi=300, bbox_inches="tight")
            fig7.savefig(FIGURES_DIR / "figure7_specification_curve.pdf", format="pdf", bbox_inches="tight")
            plt.close(fig7)
            logger.info(f"  Saved Figure 7 to {fig7_path}")
    else:
        logger.warning("Specification curve analysis produced no results")

    # 2. Bootstrap CIs
    logger.info("\n[4/10] Running bootstrap confidence intervals...")
    bootstrap_results = run_bootstrap_model(df, n_bootstrap=N_BOOTSTRAP)

    if bootstrap_results:
        bootstrap_table = format_bootstrap_table(bootstrap_results)
        bootstrap_path = SUPPLEMENTARY_DIR / "tableS1_bootstrap_ci.csv"
        bootstrap_table.to_csv(bootstrap_path, index=False)
        logger.info(f"  Saved Bootstrap CIs to {bootstrap_path}")
        print("\n" + "=" * 50)
        print("TABLE S1: Bootstrap Confidence Intervals")
        print("=" * 50)
        print(bootstrap_table.to_string(index=False))

    # 3. Leave-One-Country-Out
    logger.info("\n[5/10] Running leave-one-country-out sensitivity...")

    # Get full-sample beta for comparison
    full_beta = 0.22  # Default from Phase 3 results
    if spec_results:
        full_sample = [r for r in spec_results if r.control_set == 'full' and r.sample_filter == 'all_fines']
        if full_sample:
            full_beta = full_sample[0].beta_aggravating

    loco_results = run_leave_one_country_out(df, full_beta)

    if loco_results:
        loco_table = format_loco_table(loco_results, full_beta)
        loco_path = SUPPLEMENTARY_DIR / "tableS2_loco_sensitivity.csv"
        loco_table.to_csv(loco_path, index=False)
        logger.info(f"  Saved LOCO sensitivity to {loco_path}")
        print("\n" + "=" * 50)
        print("TABLE S2: Leave-One-Country-Out Sensitivity")
        print("=" * 50)
        print(loco_table.head(15).to_string(index=False))

    # 4. Placebo Tests
    logger.info("\n[6/10] Running placebo tests...")
    placebo_results: List[PlaceboResults] = []

    placebo_shuffle = run_placebo_article_shuffle(df, n_permutations=N_PLACEBO)
    if placebo_shuffle:
        placebo_results.append(placebo_shuffle)

    if placebo_results:
        placebo_table = format_placebo_table(placebo_results)
        placebo_path = SUPPLEMENTARY_DIR / "tableS3_placebo_tests.csv"
        placebo_table.to_csv(placebo_path, index=False)
        logger.info(f"  Saved placebo tests to {placebo_path}")
        print("\n" + "=" * 50)
        print("TABLE S3: Placebo Tests")
        print("=" * 50)
        print(placebo_table.to_string(index=False))

    # 5. Alternative Operationalizations
    logger.info("\n[7/10] Running alternative operationalizations...")
    alt_results = run_alternative_operationalizations(df)

    if alt_results:
        alt_table = format_alternative_operationalization_table(alt_results)
        alt_path = SUPPLEMENTARY_DIR / "tableS4_alternative_operationalizations.csv"
        alt_table.to_csv(alt_path, index=False)
        logger.info(f"  Saved alternative operationalizations to {alt_path}")
        print("\n" + "=" * 50)
        print("TABLE S4: Alternative Factor Operationalizations")
        print("=" * 50)
        print(alt_table.to_string(index=False))

    # 6. Mega-Fine Sensitivity Analysis (NEW)
    logger.info("\n[8/10] Running mega-fine sensitivity analysis...")
    mega_fine_results = run_mega_fine_sensitivity(df)

    if mega_fine_results:
        mega_table = format_mega_fine_sensitivity_table(mega_fine_results)
        mega_path = SUPPLEMENTARY_DIR / "tableS5_mega_fine_sensitivity.csv"
        mega_table.to_csv(mega_path, index=False)
        logger.info(f"  Saved mega-fine sensitivity to {mega_path}")
        print("\n" + "=" * 50)
        print("TABLE S5: Mega-Fine Sensitivity Analysis")
        print("=" * 50)
        print(mega_table.to_string(index=False))

    # 7. Winsorization Sensitivity Analysis (NEW)
    logger.info("\n[9/10] Running winsorization sensitivity analysis...")
    winsor_results = run_winsorization_sensitivity(df)

    if winsor_results:
        winsor_table = format_winsorization_table(winsor_results)
        winsor_path = SUPPLEMENTARY_DIR / "tableS6_winsorization_sensitivity.csv"
        winsor_table.to_csv(winsor_path, index=False)
        logger.info(f"  Saved winsorization sensitivity to {winsor_path}")
        print("\n" + "=" * 50)
        print("TABLE S6: Winsorization Sensitivity Analysis")
        print("=" * 50)
        print(winsor_table.to_string(index=False))

    # Save comprehensive summary
    logger.info("\n[10/10] Saving Phase 5 summary...")
    save_phase5_summary(spec_results, bootstrap_results, loco_results, placebo_results,
                        alt_results, mega_fine_results, winsor_results)

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5 COMPLETE")
    logger.info("=" * 70)

    print("\n" + "=" * 50)
    print("PHASE 5 SUMMARY")
    print("=" * 50)

    if spec_results:
        betas = [r.beta_aggravating for r in spec_results if not np.isnan(r.beta_aggravating)]
        p_vals = [r.p_value_aggravating for r in spec_results if not np.isnan(r.p_value_aggravating)]
        print(f"\n1. Specification Curve:")
        print(f"   Specifications: {len(spec_results)}")
        print(f"   β range: [{min(betas):.4f}, {max(betas):.4f}]")
        print(f"   % Significant: {100 * sum(1 for p in p_vals if p < 0.05) / len(p_vals):.1f}%")

    if bootstrap_results:
        print(f"\n2. Bootstrap CIs ({N_BOOTSTRAP} replicates):")
        for r in bootstrap_results[:3]:
            print(f"   {r.variable}: [{r.ci_lower_95:.4f}, {r.ci_upper_95:.4f}]")

    if loco_results:
        changes = [abs(r.change_from_full) for r in loco_results]
        print(f"\n3. Leave-One-Country-Out:")
        print(f"   Countries: {len(loco_results)}")
        print(f"   Max % change: {max(changes):.1f}%")
        print(f"   Stable: {'Yes' if max(changes) < 30 else 'No'}")

    if placebo_results:
        print(f"\n4. Placebo Tests:")
        for r in placebo_results:
            print(f"   {r.test_name}: p={r.p_value:.4f} ({'Robust' if r.p_value < 0.05 else 'Not robust'})")

    if alt_results:
        print(f"\n5. Alternative Operationalizations:")
        for r in alt_results:
            sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
            print(f"   {r.operationalization}: β={r.beta_main:.4f}{sig}")

    if mega_fine_results:
        print(f"\n6. Mega-Fine Sensitivity (Outlier Robustness):")
        for r in mega_fine_results:
            sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
            print(f"   {r.sample_description}: β={r.beta_aggravating:.4f}{sig} (N={r.n_obs})")

    if winsor_results:
        print(f"\n7. Winsorization Sensitivity (Outlier Robustness):")
        for r in winsor_results:
            sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
            print(f"   {r.winsor_level}: β={r.beta_aggravating:.4f}{sig} (N capped={r.n_affected})")

    print("\n" + "=" * 50)
    print(f"Outputs saved to:")
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Supplementary: {SUPPLEMENTARY_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
