"""Phase 2 – Descriptive Analysis: Summary Statistics and Visualizations.

This script implements Phase 2 (Descriptive Analysis) of the methodology proposal:
"Reasoning and Consistency in GDPR Enforcement: A Cross-Border Analysis"

Outputs:
- Table 1: Descriptive statistics by country
- Table 2: Article 83(2) factor frequencies and distributions
- Figure 1: Fine amount distribution by country (violin plot)
- Figure 2: Factor usage heatmap by authority
- Figure 6: Coefficient plot with 95% CIs (from Phase 3 results)

Input:  outputs/paper/data/analysis_sample.csv
        outputs/paper/tables/table3_aggregate_factors.csv (for Figure 6)
Output: outputs/paper/tables/, outputs/paper/figures/
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
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
TABLE3_PATH = Path("outputs/paper/tables/table3_aggregate_factors.csv")
TABLE4_PATH = Path("outputs/paper/tables/table4_factor_decomposition.csv")

TABLES_DIR = Path("outputs/paper/tables")
FIGURES_DIR = Path("outputs/paper/figures")
DATA_DIR = Path("outputs/paper/data")

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

FACTOR_LABELS = {
    "a59_nature_gravity_duration": "Nature/Gravity/Duration",
    "a60_intentional_negligent": "Intentional/Negligent",
    "a61_mitigate_damage_actions": "Mitigation Actions",
    "a62_technical_org_measures": "Tech/Org Measures",
    "a63_previous_infringements": "Previous Infringements",
    "a64_cooperation_authority": "Cooperation",
    "a65_data_categories_affected": "Data Categories",
    "a66_infringement_became_known": "How Infringement Known",
    "a67_prior_orders_compliance": "Prior Order Compliance",
    "a68_codes_certification": "Codes/Certification",
    "a69_other_factors": "Other Factors",
}

# Country name mapping
COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus",
    "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
    "EL": "Greece", "ES": "Spain", "FI": "Finland", "FR": "France",
    "GB": "United Kingdom", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
    "IS": "Iceland", "IT": "Italy", "LI": "Liechtenstein", "LT": "Lithuania",
    "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta", "NL": "Netherlands",
    "NO": "Norway", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_analysis_sample() -> pd.DataFrame:
    """Load the analytical sample."""
    if not ANALYSIS_SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Analysis sample not found at {ANALYSIS_SAMPLE_PATH}")
    df = pd.read_csv(ANALYSIS_SAMPLE_PATH, low_memory=False)
    logger.info(f"Loaded analysis sample: {len(df)} rows")
    return df


# -----------------------------------------------------------------------------
# Table 1: Descriptive Statistics by Country
# -----------------------------------------------------------------------------

def create_table1_country_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Table 1: Descriptive statistics by country.

    Columns:
    - Country
    - N (decisions)
    - Mean Fine (EUR)
    - Median Fine (EUR)
    - SD Fine (EUR)
    - Min Fine
    - Max Fine
    - Mean Art 83 Factors Discussed
    """
    # Get fine column
    fine_col = "fine_amount_eur_real_2025" if "fine_amount_eur_real_2025" in df.columns else "fine_amount_eur"

    rows = []

    for country in sorted(df["a1_country_code"].unique()):
        country_df = df[df["a1_country_code"] == country]
        fines = country_df[fine_col].dropna()

        # Art 83 discussed count
        discussed = country_df["art83_discussed_count"].dropna() if "art83_discussed_count" in country_df.columns else pd.Series([0])

        rows.append({
            "Country": f"{COUNTRY_NAMES.get(country, country)} ({country})",
            "N": len(country_df),
            "Mean Fine (EUR)": f"{fines.mean():,.0f}" if len(fines) > 0 else "-",
            "Median Fine (EUR)": f"{fines.median():,.0f}" if len(fines) > 0 else "-",
            "SD Fine (EUR)": f"{fines.std():,.0f}" if len(fines) > 1 else "-",
            "Min Fine (EUR)": f"{fines.min():,.0f}" if len(fines) > 0 else "-",
            "Max Fine (EUR)": f"{fines.max():,.0f}" if len(fines) > 0 else "-",
            "Mean Factors Discussed": f"{discussed.mean():.1f}" if len(discussed) > 0 else "-",
        })

    # Add total row
    fines_all = df[fine_col].dropna()
    discussed_all = df["art83_discussed_count"].dropna() if "art83_discussed_count" in df.columns else pd.Series([0])

    rows.append({
        "Country": "TOTAL",
        "N": len(df),
        "Mean Fine (EUR)": f"{fines_all.mean():,.0f}",
        "Median Fine (EUR)": f"{fines_all.median():,.0f}",
        "SD Fine (EUR)": f"{fines_all.std():,.0f}",
        "Min Fine (EUR)": f"{fines_all.min():,.0f}",
        "Max Fine (EUR)": f"{fines_all.max():,.0f}",
        "Mean Factors Discussed": f"{discussed_all.mean():.1f}",
    })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Table 2: Article 83(2) Factor Frequencies
# -----------------------------------------------------------------------------

def create_table2_factor_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Table 2: Article 83(2) factor frequencies and distributions.

    For each factor:
    - Count and % of AGGRAVATING
    - Count and % of MITIGATING
    - Count and % of NEUTRAL
    - Count and % of NOT_DISCUSSED
    """
    rows = []

    for col in ART83_FACTOR_COLS:
        if col not in df.columns:
            continue

        label = FACTOR_LABELS.get(col, col)
        values = df[col].fillna("NOT_DISCUSSED")
        total = len(values)

        agg_count = (values == "AGGRAVATING").sum()
        mit_count = (values == "MITIGATING").sum()
        neu_count = (values == "NEUTRAL").sum()
        nd_count = (values == "NOT_DISCUSSED").sum()

        rows.append({
            "Factor": label,
            "Aggravating N": agg_count,
            "Aggravating %": f"{agg_count/total*100:.1f}%",
            "Mitigating N": mit_count,
            "Mitigating %": f"{mit_count/total*100:.1f}%",
            "Neutral N": neu_count,
            "Neutral %": f"{neu_count/total*100:.1f}%",
            "Not Discussed N": nd_count,
            "Not Discussed %": f"{nd_count/total*100:.1f}%",
            "Discussion Rate": f"{(total-nd_count)/total*100:.1f}%",
        })

    # Add summary row
    factor_discussed_total = 0
    for col in ART83_FACTOR_COLS:
        if col in df.columns:
            factor_discussed_total += (df[col].fillna("NOT_DISCUSSED") != "NOT_DISCUSSED").sum()

    rows.append({
        "Factor": "--- TOTAL ---",
        "Aggravating N": sum(r["Aggravating N"] for r in rows),
        "Aggravating %": "-",
        "Mitigating N": sum(r["Mitigating N"] for r in rows),
        "Mitigating %": "-",
        "Neutral N": sum(r["Neutral N"] for r in rows),
        "Neutral %": "-",
        "Not Discussed N": sum(r["Not Discussed N"] for r in rows),
        "Not Discussed %": "-",
        "Discussion Rate": f"{factor_discussed_total / (len(df) * 11) * 100:.1f}%",
    })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Figure 1: Fine Distribution by Country (Violin Plot)
# -----------------------------------------------------------------------------

def create_figure1_fine_distribution(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Create Figure 1: Fine amount distribution by country (violin plot)."""
    if not HAS_PLOTTING:
        return None

    # Prepare data
    fine_col = "fine_amount_eur_real_2025" if "fine_amount_eur_real_2025" in df.columns else "fine_amount_eur"
    log_fine_col = "log_fine_2025" if "log_fine_2025" in df.columns else None

    if log_fine_col is None:
        df = df.copy()
        df["log_fine_2025"] = np.log1p(df[fine_col].clip(lower=0).fillna(0))
        log_fine_col = "log_fine_2025"

    # Get countries with at least 5 decisions for meaningful visualization
    country_counts = df["a1_country_code"].value_counts()
    valid_countries = country_counts[country_counts >= 5].index.tolist()

    plot_df = df[df["a1_country_code"].isin(valid_countries)].copy()

    # Sort countries by median log fine
    country_order = plot_df.groupby("a1_country_code")[log_fine_col].median().sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Top panel: Violin plot of log fine
    ax1 = axes[0]
    sns.violinplot(
        data=plot_df, x="a1_country_code", y=log_fine_col,
        order=country_order, palette="viridis", ax=ax1
    )
    ax1.set_xlabel("Country", fontsize=11)
    ax1.set_ylabel("Log Fine (EUR, 2025 Real)", fontsize=11)
    ax1.set_title("A. Distribution of Log Fine by Country", fontsize=12)
    ax1.tick_params(axis='x', rotation=45)

    # Add median line annotations
    for i, country in enumerate(country_order):
        country_data = plot_df[plot_df["a1_country_code"] == country][log_fine_col]
        median_val = country_data.median()
        ax1.hlines(median_val, i-0.4, i+0.4, color='red', linewidth=2, alpha=0.7)

    # Bottom panel: Box plot with jittered points
    ax2 = axes[1]

    # Box plot
    sns.boxplot(
        data=plot_df, x="a1_country_code", y=log_fine_col,
        order=country_order, palette="viridis", ax=ax2, showfliers=False
    )

    # Add jittered points
    for i, country in enumerate(country_order):
        country_data = plot_df[plot_df["a1_country_code"] == country][log_fine_col].values
        x_jitter = np.random.uniform(-0.2, 0.2, len(country_data)) + i
        ax2.scatter(x_jitter, country_data, alpha=0.4, s=20, color='black', zorder=10)

    ax2.set_xlabel("Country", fontsize=11)
    ax2.set_ylabel("Log Fine (EUR, 2025 Real)", fontsize=11)
    ax2.set_title("B. Log Fine Box Plot with Individual Decisions", fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    # Add N annotations below x-axis
    for i, country in enumerate(country_order):
        n = len(plot_df[plot_df["a1_country_code"] == country])
        ax2.annotate(f"n={n}", xy=(i, ax2.get_ylim()[0]), xytext=(0, -25),
                     textcoords="offset points", ha='center', fontsize=8)

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Figure 2: Factor Usage Heatmap by Authority
# -----------------------------------------------------------------------------

def create_figure2_factor_heatmap(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Create Figure 2: Factor usage heatmap by authority."""
    if not HAS_PLOTTING:
        return None

    # Get authorities with at least 10 decisions
    auth_col = "authority_name_norm" if "authority_name_norm" in df.columns else "a2_authority_name"
    auth_counts = df[auth_col].value_counts()
    valid_auths = auth_counts[auth_counts >= 10].index.tolist()

    if len(valid_auths) < 3:
        logger.warning("Insufficient authorities for heatmap")
        return None

    # Limit to top 20 authorities
    valid_auths = valid_auths[:20]

    # Compute factor usage rates for each authority
    heatmap_data = []

    for auth in valid_auths:
        auth_df = df[df[auth_col] == auth]
        n = len(auth_df)

        row = {"Authority": auth[:35] + "..." if len(auth) > 35 else auth, "N": n}

        for col in ART83_FACTOR_COLS:
            if col not in auth_df.columns:
                row[FACTOR_LABELS.get(col, col)] = 0
            else:
                # Discussion rate (non-NOT_DISCUSSED)
                discussed_rate = (auth_df[col].fillna("NOT_DISCUSSED") != "NOT_DISCUSSED").mean()
                row[FACTOR_LABELS.get(col, col)] = discussed_rate * 100

        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df = heatmap_df.set_index("Authority")
    n_values = heatmap_df["N"]
    heatmap_df = heatmap_df.drop(columns=["N"])

    # Sort by total factor usage
    heatmap_df["_total"] = heatmap_df.sum(axis=1)
    heatmap_df = heatmap_df.sort_values("_total", ascending=False)
    heatmap_df = heatmap_df.drop(columns=["_total"])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(
        heatmap_df, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={'label': 'Discussion Rate (%)'}
    )

    ax.set_xlabel("Article 83(2) Factor", fontsize=11)
    ax.set_ylabel("Data Protection Authority", fontsize=11)
    ax.set_title("Factor Discussion Rates by Authority\n(% of decisions where factor was discussed)", fontsize=12)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Figure 6: Coefficient Plot with 95% CIs
# -----------------------------------------------------------------------------

def create_figure6_coefficient_plot(table4_path: Path) -> Optional[plt.Figure]:
    """Create Figure 6: Coefficient plot with 95% CIs from Table 4."""
    if not HAS_PLOTTING:
        return None

    if not table4_path.exists():
        logger.warning(f"Table 4 not found at {table4_path}")
        return None

    # Load Table 4 (factor decomposition results)
    table4 = pd.read_csv(table4_path)

    if len(table4) == 0:
        return None

    # Parse coefficients and CIs
    # Expected columns: Article 83(2) Factor, Coefficient, Std. Error, 95% CI

    plot_data = []

    for _, row in table4.iterrows():
        factor = row.get("Article 83(2) Factor", "")
        coef_str = str(row.get("Coefficient", "0"))
        ci_str = str(row.get("95% CI", "[0, 0]"))

        # Extract numeric coefficient (remove significance stars)
        import re
        coef_match = re.match(r"([-+]?\d*\.?\d+)", coef_str)
        if coef_match:
            coef = float(coef_match.group(1))
        else:
            continue

        # Extract CI bounds
        ci_match = re.findall(r"[-+]?\d*\.?\d+", ci_str)
        if len(ci_match) >= 2:
            ci_lower = float(ci_match[0])
            ci_upper = float(ci_match[1])
        else:
            ci_lower = coef - 0.2
            ci_upper = coef + 0.2

        # Check significance
        is_significant = "*" in coef_str

        plot_data.append({
            "Factor": factor,
            "Coefficient": coef,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper,
            "Significant": is_significant,
        })

    if not plot_data:
        return None

    plot_df = pd.DataFrame(plot_data)

    # Sort by coefficient magnitude
    plot_df = plot_df.sort_values("Coefficient", ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    y_positions = range(len(plot_df))
    colors = ['darkgreen' if sig else 'gray' for sig in plot_df["Significant"]]

    # Plot error bars
    ax.hlines(y_positions, plot_df["CI_Lower"], plot_df["CI_Upper"],
              colors=colors, linewidth=2, alpha=0.7)

    # Plot points
    ax.scatter(plot_df["Coefficient"], y_positions, c=colors, s=100, zorder=10)

    # Add zero reference line
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["Factor"])
    ax.set_xlabel("Coefficient (Effect on Log Fine)", fontsize=11)
    ax.set_ylabel("Article 83(2) Factor", fontsize=11)
    ax.set_title("Factor Effects on Fine Magnitude\n(Mixed-Effects Model with 95% CIs)", fontsize=12)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen',
               markersize=10, label='Significant (p<0.05)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='Not Significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    """Create output directories if they don't exist."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Execute Phase 2: Descriptive Analysis."""
    logger.info("=" * 70)
    logger.info("PHASE 2: DESCRIPTIVE ANALYSIS")
    logger.info("=" * 70)

    ensure_output_dirs()

    # Load data
    logger.info("\n[1/6] Loading analysis sample...")
    df = load_analysis_sample()

    # Table 1: Country statistics
    logger.info("\n[2/6] Creating Table 1: Country descriptive statistics...")
    table1 = create_table1_country_stats(df)
    table1_path = TABLES_DIR / "table1_country_statistics.csv"
    table1.to_csv(table1_path, index=False)
    logger.info(f"  Saved Table 1 to {table1_path}")
    print("\n" + "=" * 50)
    print("TABLE 1: Descriptive Statistics by Country")
    print("=" * 50)
    print(table1.to_string(index=False))

    # Table 2: Factor frequencies
    logger.info("\n[3/6] Creating Table 2: Article 83(2) factor frequencies...")
    table2 = create_table2_factor_frequencies(df)
    table2_path = TABLES_DIR / "table2_factor_frequencies.csv"
    table2.to_csv(table2_path, index=False)
    logger.info(f"  Saved Table 2 to {table2_path}")
    print("\n" + "=" * 50)
    print("TABLE 2: Article 83(2) Factor Frequencies")
    print("=" * 50)
    print(table2.to_string(index=False))

    # Figure 1: Fine distribution by country
    logger.info("\n[4/6] Creating Figure 1: Fine distribution by country...")
    fig1 = create_figure1_fine_distribution(df)
    if fig1:
        fig1_path = FIGURES_DIR / "figure1_fine_distribution.png"
        fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
        fig1.savefig(FIGURES_DIR / "figure1_fine_distribution.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig1)
        logger.info(f"  Saved Figure 1 to {fig1_path}")

    # Figure 2: Factor usage heatmap
    logger.info("\n[5/6] Creating Figure 2: Factor usage heatmap...")
    fig2 = create_figure2_factor_heatmap(df)
    if fig2:
        fig2_path = FIGURES_DIR / "figure2_factor_heatmap.png"
        fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
        fig2.savefig(FIGURES_DIR / "figure2_factor_heatmap.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig2)
        logger.info(f"  Saved Figure 2 to {fig2_path}")

    # Figure 6: Coefficient plot
    logger.info("\n[6/6] Creating Figure 6: Coefficient plot...")
    fig6 = create_figure6_coefficient_plot(TABLE4_PATH)
    if fig6:
        fig6_path = FIGURES_DIR / "figure6_coefficient_plot.png"
        fig6.savefig(fig6_path, dpi=300, bbox_inches="tight")
        fig6.savefig(FIGURES_DIR / "figure6_coefficient_plot.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig6)
        logger.info(f"  Saved Figure 6 to {fig6_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 COMPLETE")
    logger.info("=" * 70)

    print("\n" + "=" * 50)
    print("PHASE 2 SUMMARY")
    print("=" * 50)
    print(f"Analytical sample: {len(df)} decisions")
    print(f"Countries: {df['a1_country_code'].nunique()}")

    fine_col = "fine_amount_eur_real_2025" if "fine_amount_eur_real_2025" in df.columns else "fine_amount_eur"
    print(f"\nFine Statistics:")
    print(f"  Mean: EUR {df[fine_col].mean():,.0f}")
    print(f"  Median: EUR {df[fine_col].median():,.0f}")
    print(f"  Max: EUR {df[fine_col].max():,.0f}")

    if "art83_discussed_count" in df.columns:
        print(f"\nFactor Discussion:")
        print(f"  Mean factors discussed: {df['art83_discussed_count'].mean():.1f} of 11")
        print(f"  Max factors discussed: {df['art83_discussed_count'].max():.0f}")

    print("\n" + "=" * 50)
    print(f"Outputs saved to:")
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
