"""Phase 2 – Paper Descriptive Analysis: Summary Statistics and Visualizations.

This script implements Phase 2 (Descriptive Analysis) of the methodology proposal:
"Reasoning and Consistency in GDPR Enforcement: A Cross-Border Analysis"

Tasks performed:
1. Generate Table 1: Sample characteristics by country
2. Generate Table 2: Article 83(2) factor frequencies and distributions
3. Generate Figure 1: Fine amount distribution by country (violin plot)
4. Generate Figure 2: Factor usage heatmap by authority
5. Compute additional descriptive statistics for paper

Input:  outputs/paper/data/analysis_sample.csv
        outputs/paper/data/authority_systematicity.csv
        outputs/paper/data/cohort_membership.csv
Output: outputs/paper/tables/
        - table1_country_characteristics.csv      (country-level summary)
        - table1_country_characteristics.tex      (LaTeX format)
        - table2_factor_frequencies.csv           (factor usage statistics)
        - table2_factor_frequencies.tex           (LaTeX format)
        outputs/paper/figures/
        - figure1_fine_distribution_country.png   (violin plot)
        - figure1_fine_distribution_country.pdf   (vector format)
        - figure2_factor_heatmap.png              (authority heatmap)
        - figure2_factor_heatmap.pdf              (vector format)
        outputs/paper/data/
        - descriptive_stats_summary.txt           (comprehensive summary)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SAMPLE_PATH = Path("outputs/paper/data/analysis_sample.csv")
SYSTEMATICITY_PATH = Path("outputs/paper/data/authority_systematicity.csv")
COHORT_PATH = Path("outputs/paper/data/cohort_membership.csv")
REGION_MAP_PATH = Path("raw_data/reference/region_map.csv")

OUTPUT_DIR_TABLES = Path("outputs/paper/tables")
OUTPUT_DIR_FIGURES = Path("outputs/paper/figures")
OUTPUT_DIR_DATA = Path("outputs/paper/data")

# Random seed for reproducibility
RANDOM_SEED = 42

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = ['png', 'pdf']

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

# Human-readable factor names for display
FACTOR_DISPLAY_NAMES = {
    "a59_nature_gravity_duration": "Nature, gravity, duration",
    "a60_intentional_negligent": "Intent/negligence",
    "a61_mitigate_damage_actions": "Damage mitigation",
    "a62_technical_org_measures": "Tech/org measures",
    "a63_previous_infringements": "Previous infringements",
    "a64_cooperation_authority": "Cooperation",
    "a65_data_categories_affected": "Data categories",
    "a66_infringement_became_known": "How discovered",
    "a67_prior_orders_compliance": "Prior orders",
    "a68_codes_certification": "Codes/certification",
    "a69_other_factors": "Other factors",
}

# Region display order
REGION_ORDER = ["DACH", "Nordics", "Benelux", "British-Irish", "Southern", "CEE", "EEA-Non-EU"]

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_analytical_sample() -> pd.DataFrame:
    """Load the Phase 1 analytical sample."""
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Analytical sample not found at {SAMPLE_PATH}. "
            "Run Phase 1 (6_paper_data_preparation.py) first."
        )
    df = pd.read_csv(SAMPLE_PATH, low_memory=False)
    logger.info(f"Loaded analytical sample: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_systematicity() -> pd.DataFrame:
    """Load authority systematicity indices."""
    if not SYSTEMATICITY_PATH.exists():
        logger.warning(f"Systematicity file not found at {SYSTEMATICITY_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(SYSTEMATICITY_PATH)
    logger.info(f"Loaded systematicity indices: {len(df)} authorities")
    return df


def load_region_map() -> Dict[str, str]:
    """Load country-to-region mapping."""
    if not REGION_MAP_PATH.exists():
        logger.warning(f"Region map not found at {REGION_MAP_PATH}")
        return {}
    region_df = pd.read_csv(REGION_MAP_PATH)
    return dict(zip(region_df["country_code"], region_df["region"]))


# -----------------------------------------------------------------------------
# Table 1: Country Characteristics
# -----------------------------------------------------------------------------

def generate_table1_country_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 1: Sample characteristics by country.

    Columns:
    - Country code
    - Region
    - N (cases)
    - Fine Mean (EUR)
    - Fine Median (EUR)
    - Fine SD
    - Log Fine Mean
    - Log Fine SD
    - Aggravating Count (mean)
    - Mitigating Count (mean)
    - Systematicity Coverage
    """
    region_map = load_region_map()

    records = []
    for country, group in df.groupby("a1_country_code"):
        if pd.isna(country):
            continue

        fine_col = "fine_amount_eur_real_2025"
        if fine_col not in group.columns:
            fine_col = "fine_amount_eur"

        log_fine_col = "log_fine_2025"

        # Compute statistics
        n_cases = len(group)
        fine_mean = group[fine_col].mean()
        fine_median = group[fine_col].median()
        fine_sd = group[fine_col].std()
        fine_min = group[fine_col].min()
        fine_max = group[fine_col].max()

        log_fine_mean = group[log_fine_col].mean() if log_fine_col in group.columns else np.nan
        log_fine_sd = group[log_fine_col].std() if log_fine_col in group.columns else np.nan

        # Article 83(2) factor counts
        agg_count_mean = group["art83_aggravating_count"].mean() if "art83_aggravating_count" in group.columns else np.nan
        mit_count_mean = group["art83_mitigating_count"].mean() if "art83_mitigating_count" in group.columns else np.nan
        discussed_count_mean = group["art83_discussed_count"].mean() if "art83_discussed_count" in group.columns else np.nan

        # Defendant characteristics
        pct_private = (group["a8_defendant_class"] == "PRIVATE").mean() * 100
        pct_public = (group["a8_defendant_class"] == "PUBLIC").mean() * 100

        # Enterprise size distribution
        pct_sme = (group["a9_enterprise_size"] == "SME").mean() * 100
        pct_large = (group["a9_enterprise_size"].isin(["LARGE", "VERY_LARGE"])).mean() * 100

        # Year range
        year_min = int(group["decision_year"].min()) if pd.notna(group["decision_year"].min()) else np.nan
        year_max = int(group["decision_year"].max()) if pd.notna(group["decision_year"].max()) else np.nan

        records.append({
            "country_code": country,
            "region": region_map.get(country, "Unknown"),
            "n_cases": n_cases,
            "fine_mean_eur": fine_mean,
            "fine_median_eur": fine_median,
            "fine_sd_eur": fine_sd,
            "fine_min_eur": fine_min,
            "fine_max_eur": fine_max,
            "log_fine_mean": log_fine_mean,
            "log_fine_sd": log_fine_sd,
            "art83_agg_mean": agg_count_mean,
            "art83_mit_mean": mit_count_mean,
            "art83_discussed_mean": discussed_count_mean,
            "pct_private": pct_private,
            "pct_public": pct_public,
            "pct_sme": pct_sme,
            "pct_large": pct_large,
            "year_min": year_min,
            "year_max": year_max,
        })

    table1 = pd.DataFrame.from_records(records)

    # Sort by region then by n_cases
    region_order_map = {r: i for i, r in enumerate(REGION_ORDER)}
    table1["region_order"] = table1["region"].map(region_order_map).fillna(99)
    table1 = table1.sort_values(["region_order", "n_cases"], ascending=[True, False])
    table1 = table1.drop(columns=["region_order"])

    logger.info(f"Generated Table 1: {len(table1)} countries")
    return table1


def table1_to_latex(df: pd.DataFrame) -> str:
    """Convert Table 1 to LaTeX format for paper."""
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sample Characteristics by Country}",
        r"\label{tab:country_characteristics}",
        r"\small",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Country & Region & N & Median Fine & Mean Fine & SD & Agg. & Mit. & Disc. \\",
        r" & & & (EUR) & (EUR) & (EUR) & (mean) & (mean) & (mean) \\",
        r"\midrule",
    ]

    current_region = None
    for _, row in df.iterrows():
        if row["region"] != current_region:
            current_region = row["region"]
            if latex_lines[-1] != r"\midrule":
                latex_lines.append(r"\midrule")

        # Format numbers
        median_fmt = f"{row['fine_median_eur']:,.0f}" if pd.notna(row['fine_median_eur']) else "--"
        mean_fmt = f"{row['fine_mean_eur']:,.0f}" if pd.notna(row['fine_mean_eur']) else "--"
        sd_fmt = f"{row['fine_sd_eur']:,.0f}" if pd.notna(row['fine_sd_eur']) else "--"
        agg_fmt = f"{row['art83_agg_mean']:.2f}" if pd.notna(row['art83_agg_mean']) else "--"
        mit_fmt = f"{row['art83_mit_mean']:.2f}" if pd.notna(row['art83_mit_mean']) else "--"
        disc_fmt = f"{row['art83_discussed_mean']:.2f}" if pd.notna(row['art83_discussed_mean']) else "--"

        line = (f"{row['country_code']} & {row['region']} & {row['n_cases']} & "
                f"{median_fmt} & {mean_fmt} & {sd_fmt} & "
                f"{agg_fmt} & {mit_fmt} & {disc_fmt} \\\\")
        latex_lines.append(line)

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Note: Fine amounts in 2025 EUR (real). Agg. = Art. 83(2) aggravating factors; ",
        r"Mit. = mitigating factors; Disc. = factors discussed (out of 11).",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(latex_lines)


# -----------------------------------------------------------------------------
# Table 2: Article 83(2) Factor Frequencies
# -----------------------------------------------------------------------------

def generate_table2_factor_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 2: Article 83(2) factor frequencies and distributions.

    For each factor:
    - N discussed (not NOT_DISCUSSED)
    - % discussed
    - % AGGRAVATING
    - % MITIGATING
    - % NEUTRAL
    """
    records = []
    total_cases = len(df)

    for col in ART83_FACTOR_COLS:
        if col not in df.columns:
            continue

        values = df[col].fillna("NOT_DISCUSSED")

        # Counts
        n_discussed = (values != "NOT_DISCUSSED").sum()
        n_aggravating = (values == "AGGRAVATING").sum()
        n_mitigating = (values == "MITIGATING").sum()
        n_neutral = (values == "NEUTRAL").sum()
        n_not_discussed = (values == "NOT_DISCUSSED").sum()

        # Percentages (of total)
        pct_discussed = (n_discussed / total_cases) * 100
        pct_aggravating = (n_aggravating / total_cases) * 100
        pct_mitigating = (n_mitigating / total_cases) * 100
        pct_neutral = (n_neutral / total_cases) * 100

        # Percentages among discussed
        pct_agg_of_disc = (n_aggravating / n_discussed * 100) if n_discussed > 0 else np.nan
        pct_mit_of_disc = (n_mitigating / n_discussed * 100) if n_discussed > 0 else np.nan
        pct_neu_of_disc = (n_neutral / n_discussed * 100) if n_discussed > 0 else np.nan

        records.append({
            "factor": col,
            "factor_display": FACTOR_DISPLAY_NAMES.get(col, col),
            "n_total": total_cases,
            "n_discussed": n_discussed,
            "n_aggravating": n_aggravating,
            "n_mitigating": n_mitigating,
            "n_neutral": n_neutral,
            "n_not_discussed": n_not_discussed,
            "pct_discussed": pct_discussed,
            "pct_aggravating": pct_aggravating,
            "pct_mitigating": pct_mitigating,
            "pct_neutral": pct_neutral,
            "pct_agg_of_discussed": pct_agg_of_disc,
            "pct_mit_of_discussed": pct_mit_of_disc,
            "pct_neu_of_discussed": pct_neu_of_disc,
        })

    table2 = pd.DataFrame.from_records(records)
    table2 = table2.sort_values("pct_discussed", ascending=False)

    logger.info(f"Generated Table 2: {len(table2)} factors")
    return table2


def table2_to_latex(df: pd.DataFrame) -> str:
    """Convert Table 2 to LaTeX format for paper."""
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Article 83(2) Factor Usage Frequencies}",
        r"\label{tab:factor_frequencies}",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Factor & \multicolumn{2}{c}{Discussed} & \multicolumn{2}{c}{Aggravating} & \multicolumn{2}{c}{Mitigating} & Neutral \\",
        r" & N & \% & N & \% & N & \% & \% \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        line = (f"{row['factor_display']} & "
                f"{row['n_discussed']:.0f} & {row['pct_discussed']:.1f} & "
                f"{row['n_aggravating']:.0f} & {row['pct_aggravating']:.1f} & "
                f"{row['n_mitigating']:.0f} & {row['pct_mitigating']:.1f} & "
                f"{row['pct_neutral']:.1f} \\\\")
        latex_lines.append(line)

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Note: N = 528 GDPR enforcement decisions with fines imposed. ",
        r"Percentages are of total sample. Factors are from Article 83(2) GDPR.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    return "\n".join(latex_lines)


# -----------------------------------------------------------------------------
# Figure 1: Fine Distribution by Country (Violin Plot)
# -----------------------------------------------------------------------------

def generate_figure1_violin_plot(df: pd.DataFrame) -> plt.Figure:
    """
    Generate Figure 1: Fine amount distribution by country (violin plot).

    Uses log-transformed fine amounts for visualization.
    Countries ordered by region, then by median fine.
    """
    region_map = load_region_map()

    # Prepare data
    plot_df = df.copy()
    plot_df["region"] = plot_df["a1_country_code"].map(region_map).fillna("Unknown")
    plot_df["log_fine"] = plot_df["log_fine_2025"]

    # Filter to countries with sufficient data (>=5 cases)
    country_counts = plot_df["a1_country_code"].value_counts()
    valid_countries = country_counts[country_counts >= 5].index
    plot_df = plot_df[plot_df["a1_country_code"].isin(valid_countries)]

    # Order countries by region then median fine
    country_order = (plot_df.groupby("a1_country_code")["log_fine"]
                     .median()
                     .sort_values(ascending=False)
                     .index.tolist())

    # Create region ordering
    region_order_map = {r: i for i, r in enumerate(REGION_ORDER)}
    country_region = plot_df.groupby("a1_country_code")["region"].first()
    country_region_order = country_region.map(region_order_map).fillna(99)

    # Sort by region then median
    country_df = pd.DataFrame({
        "country": country_region.index,
        "region_order": country_region_order.values,
        "median_fine": plot_df.groupby("a1_country_code")["log_fine"].median().values
    })
    country_df = country_df.sort_values(["region_order", "median_fine"], ascending=[True, False])
    country_order = country_df["country"].tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color by region
    region_colors = {
        "DACH": "#1f77b4",
        "Nordics": "#2ca02c",
        "Benelux": "#ff7f0e",
        "British-Irish": "#d62728",
        "Southern": "#9467bd",
        "CEE": "#8c564b",
        "EEA-Non-EU": "#7f7f7f",
        "Unknown": "#bcbd22",
    }

    palette = [region_colors.get(country_region.get(c, "Unknown"), "#bcbd22")
               for c in country_order]

    # Violin plot
    sns.violinplot(
        data=plot_df,
        x="a1_country_code",
        y="log_fine",
        order=country_order,
        palette=palette,
        inner="box",
        cut=0,
        ax=ax
    )

    # Add sample sizes
    for i, country in enumerate(country_order):
        n = (plot_df["a1_country_code"] == country).sum()
        ax.text(i, ax.get_ylim()[0] - 0.5, f"n={n}",
                ha='center', va='top', fontsize=8, color='gray')

    # Labels and formatting
    ax.set_xlabel("Country", fontsize=12)
    ax.set_ylabel("Log Fine Amount (EUR 2025, real)", fontsize=12)
    ax.set_title("Figure 1: Fine Amount Distribution by Country", fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    # Add legend for regions
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=region)
                      for region, color in region_colors.items()
                      if region in plot_df["region"].unique()]
    ax.legend(handles=legend_elements, title="Region", loc='upper right', fontsize=9)

    # Add horizontal grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    logger.info(f"Generated Figure 1: {len(country_order)} countries")
    return fig


# -----------------------------------------------------------------------------
# Figure 2: Factor Usage Heatmap by Authority
# -----------------------------------------------------------------------------

def generate_figure2_heatmap(df: pd.DataFrame, systematicity_df: pd.DataFrame) -> plt.Figure:
    """
    Generate Figure 2: Factor usage heatmap by authority.

    Rows: Authorities (sorted by systematicity index)
    Columns: Article 83(2) factors
    Values: % of cases where factor was discussed (any value except NOT_DISCUSSED)
    """
    # Filter to authorities with sufficient data
    auth_counts = df["a2_authority_name"].value_counts()
    valid_authorities = auth_counts[auth_counts >= 10].index
    plot_df = df[df["a2_authority_name"].isin(valid_authorities)].copy()

    # Build usage matrix
    factor_cols = [c for c in ART83_FACTOR_COLS if c in df.columns]

    matrix_data = []
    auth_order = []

    # Order by systematicity if available
    if len(systematicity_df) > 0:
        sys_order = systematicity_df.sort_values("systematicity", ascending=False)["authority_id"].tolist()
        # Match authority names
        auth_order = [a for a in sys_order if a in valid_authorities]
        # Add any missing authorities
        auth_order.extend([a for a in valid_authorities if a not in auth_order])
    else:
        auth_order = list(valid_authorities)

    for auth in auth_order:
        auth_df = plot_df[plot_df["a2_authority_name"] == auth]
        n_total = len(auth_df)

        if n_total == 0:
            continue

        row = {"authority": auth, "n_cases": n_total}
        for col in factor_cols:
            n_discussed = (auth_df[col] != "NOT_DISCUSSED").sum()
            pct_discussed = (n_discussed / n_total) * 100
            row[FACTOR_DISPLAY_NAMES.get(col, col)] = pct_discussed

        matrix_data.append(row)

    if not matrix_data:
        logger.warning("No authorities with sufficient data for heatmap")
        return plt.figure()

    matrix_df = pd.DataFrame(matrix_data)
    matrix_df = matrix_df.set_index("authority")

    # Select only factor columns for heatmap
    factor_display_names = [FACTOR_DISPLAY_NAMES.get(c, c) for c in factor_cols]
    heatmap_data = matrix_df[factor_display_names]

    # Truncate authority names for display
    heatmap_data.index = [a[:40] + "..." if len(a) > 40 else a for a in heatmap_data.index]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(heatmap_data) * 0.4)))

    # Heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        cbar_kws={"label": "% Cases Discussed", "shrink": 0.8},
        ax=ax
    )

    ax.set_xlabel("Article 83(2) Factor", fontsize=12)
    ax.set_ylabel("Authority", fontsize=12)
    ax.set_title("Figure 2: Factor Usage by Authority (% Discussed)", fontsize=14, fontweight='bold')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    logger.info(f"Generated Figure 2: {len(heatmap_data)} authorities × {len(factor_cols)} factors")
    return fig


# -----------------------------------------------------------------------------
# Additional Descriptive Statistics
# -----------------------------------------------------------------------------

def generate_descriptive_summary(df: pd.DataFrame, table1: pd.DataFrame,
                                  table2: pd.DataFrame, systematicity_df: pd.DataFrame) -> str:
    """Generate comprehensive descriptive statistics summary for paper."""
    lines = [
        "=" * 70,
        "GDPR ENFORCEMENT PAPER - PHASE 2 DESCRIPTIVE ANALYSIS",
        "=" * 70,
        "",
        "1. SAMPLE OVERVIEW",
        "-" * 40,
        f"Total analytical sample: N = {len(df):,}",
        f"Countries represented: {df['a1_country_code'].nunique()}",
        f"Unique authorities: {df['a2_authority_name'].nunique()}",
        f"Year range: {int(df['decision_year'].min())}-{int(df['decision_year'].max())}",
        "",
    ]

    # Fine statistics
    fine_col = "fine_amount_eur_real_2025" if "fine_amount_eur_real_2025" in df.columns else "fine_amount_eur"
    lines.extend([
        "2. FINE AMOUNT STATISTICS (EUR 2025 Real)",
        "-" * 40,
        f"Mean:   €{df[fine_col].mean():,.2f}",
        f"Median: €{df[fine_col].median():,.2f}",
        f"SD:     €{df[fine_col].std():,.2f}",
        f"Min:    €{df[fine_col].min():,.2f}",
        f"Max:    €{df[fine_col].max():,.2f}",
        f"IQR:    €{df[fine_col].quantile(0.25):,.2f} - €{df[fine_col].quantile(0.75):,.2f}",
        "",
    ])

    # Log fine statistics
    if "log_fine_2025" in df.columns:
        lines.extend([
            "3. LOG FINE STATISTICS",
            "-" * 40,
            f"Mean:   {df['log_fine_2025'].mean():.3f}",
            f"Median: {df['log_fine_2025'].median():.3f}",
            f"SD:     {df['log_fine_2025'].std():.3f}",
            f"Range:  {df['log_fine_2025'].min():.3f} - {df['log_fine_2025'].max():.3f}",
            "",
        ])

    # Article 83(2) factor summary
    lines.extend([
        "4. ARTICLE 83(2) FACTOR USAGE SUMMARY",
        "-" * 40,
    ])

    if "art83_discussed_count" in df.columns:
        lines.extend([
            f"Mean factors discussed per case: {df['art83_discussed_count'].mean():.2f} (of 11)",
            f"Median factors discussed: {df['art83_discussed_count'].median():.1f}",
            f"Cases with ≥5 factors discussed: {(df['art83_discussed_count'] >= 5).sum()} ({(df['art83_discussed_count'] >= 5).mean()*100:.1f}%)",
            f"Cases with 0 factors discussed: {(df['art83_discussed_count'] == 0).sum()} ({(df['art83_discussed_count'] == 0).mean()*100:.1f}%)",
            "",
        ])

    if "art83_aggravating_count" in df.columns:
        lines.extend([
            f"Mean aggravating factors: {df['art83_aggravating_count'].mean():.2f}",
            f"Mean mitigating factors: {df['art83_mitigating_count'].mean():.2f}",
            f"Mean balance score: {df['art83_balance_score'].mean():.2f}",
            "",
        ])

    # Most/least discussed factors
    if len(table2) > 0:
        lines.extend([
            "5. FACTOR DISCUSSION RATES",
            "-" * 40,
            "Most discussed factors:",
        ])
        for _, row in table2.head(3).iterrows():
            lines.append(f"  - {row['factor_display']}: {row['pct_discussed']:.1f}%")

        lines.append("\nLeast discussed factors:")
        for _, row in table2.tail(3).iterrows():
            lines.append(f"  - {row['factor_display']}: {row['pct_discussed']:.1f}%")
        lines.append("")

    # Defendant characteristics
    lines.extend([
        "6. DEFENDANT CHARACTERISTICS",
        "-" * 40,
    ])

    class_dist = df["a8_defendant_class"].value_counts(normalize=True) * 100
    lines.append("Defendant Class:")
    for cls, pct in class_dist.head(5).items():
        lines.append(f"  - {cls}: {pct:.1f}%")

    size_dist = df["a9_enterprise_size"].value_counts(normalize=True) * 100
    lines.append("\nEnterprise Size:")
    for size, pct in size_dist.head(5).items():
        lines.append(f"  - {size}: {pct:.1f}%")

    # Sector distribution
    if "a12_sector" in df.columns:
        sector_dist = df["a12_sector"].value_counts(normalize=True) * 100
        lines.append("\nTop 5 Sectors:")
        for sector, pct in sector_dist.head(5).items():
            lines.append(f"  - {sector}: {pct:.1f}%")

    lines.append("")

    # Geographic distribution
    lines.extend([
        "7. GEOGRAPHIC DISTRIBUTION",
        "-" * 40,
    ])

    region_map = load_region_map()
    df["_region"] = df["a1_country_code"].map(region_map).fillna("Unknown")
    region_dist = df["_region"].value_counts()
    lines.append("By Region:")
    for region, count in region_dist.items():
        pct = count / len(df) * 100
        lines.append(f"  - {region}: {count} ({pct:.1f}%)")

    lines.append("\nTop 5 Countries:")
    country_dist = df["a1_country_code"].value_counts()
    for country, count in country_dist.head(5).items():
        pct = count / len(df) * 100
        lines.append(f"  - {country}: {count} ({pct:.1f}%)")

    lines.append("")

    # Systematicity summary
    if len(systematicity_df) > 0:
        lines.extend([
            "8. AUTHORITY SYSTEMATICITY",
            "-" * 40,
            f"Authorities indexed (≥10 decisions): {len(systematicity_df)}",
            f"Mean systematicity: {systematicity_df['systematicity'].mean():.4f}",
            f"Median systematicity: {systematicity_df['systematicity'].median():.4f}",
            f"Systematicity range: {systematicity_df['systematicity'].min():.4f} - {systematicity_df['systematicity'].max():.4f}",
            "",
            "Top 5 by Systematicity:",
        ])
        for _, row in systematicity_df.head(5).iterrows():
            lines.append(f"  - {row['authority_id'][:40]}: {row['systematicity']:.4f} (n={row['n_decisions']})")
        lines.append("")

    # Cross-border eligibility
    if "cross_border_eligible" in df.columns:
        lines.extend([
            "9. CROSS-BORDER ANALYSIS ELIGIBILITY",
            "-" * 40,
            f"Cases cross-border eligible: {df['cross_border_eligible'].sum()} ({df['cross_border_eligible'].mean()*100:.1f}%)",
        ])

        if "article_set_key" in df.columns:
            cohort_counts = df.groupby("article_set_key").size()
            lines.append(f"Unique article cohorts: {len(cohort_counts)}")
            lines.append(f"Cohorts with ≥5 cases: {(cohort_counts >= 5).sum()}")

    lines.extend([
        "",
        "=" * 70,
        "Generated by 7_paper_descriptive_analysis.py",
        "=" * 70,
    ])

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Output Functions
# -----------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    """Create output directories if they don't exist."""
    OUTPUT_DIR_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_DATA.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directories ready: tables/, figures/, data/")


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure in multiple formats."""
    for fmt in FIGURE_FORMAT:
        path = OUTPUT_DIR_FIGURES / f"{name}.{fmt}"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight', format=fmt)
        logger.info(f"  - Saved {path}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main() -> None:
    """Execute Phase 2: Descriptive Analysis."""
    logger.info("=" * 70)
    logger.info("PHASE 2: PAPER DESCRIPTIVE ANALYSIS")
    logger.info("=" * 70)

    # Set random seed and style
    np.random.seed(RANDOM_SEED)
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Ensure output directories exist
    ensure_output_dirs()

    # Load data
    logger.info("\n[1/7] Loading data...")
    df = load_analytical_sample()
    systematicity_df = load_systematicity()

    # Generate Table 1
    logger.info("\n[2/7] Generating Table 1: Country characteristics...")
    table1 = generate_table1_country_characteristics(df)
    table1.to_csv(OUTPUT_DIR_TABLES / "table1_country_characteristics.csv", index=False)
    with open(OUTPUT_DIR_TABLES / "table1_country_characteristics.tex", "w") as f:
        f.write(table1_to_latex(table1))
    logger.info(f"  - table1_country_characteristics.csv: {len(table1)} countries")

    # Generate Table 2
    logger.info("\n[3/7] Generating Table 2: Factor frequencies...")
    table2 = generate_table2_factor_frequencies(df)
    table2.to_csv(OUTPUT_DIR_TABLES / "table2_factor_frequencies.csv", index=False)
    with open(OUTPUT_DIR_TABLES / "table2_factor_frequencies.tex", "w") as f:
        f.write(table2_to_latex(table2))
    logger.info(f"  - table2_factor_frequencies.csv: {len(table2)} factors")

    # Generate Figure 1
    logger.info("\n[4/7] Generating Figure 1: Fine distribution violin plot...")
    fig1 = generate_figure1_violin_plot(df)
    save_figure(fig1, "figure1_fine_distribution_country")
    plt.close(fig1)

    # Generate Figure 2
    logger.info("\n[5/7] Generating Figure 2: Factor usage heatmap...")
    fig2 = generate_figure2_heatmap(df, systematicity_df)
    save_figure(fig2, "figure2_factor_heatmap")
    plt.close(fig2)

    # Generate descriptive summary
    logger.info("\n[6/7] Generating descriptive statistics summary...")
    summary = generate_descriptive_summary(df, table1, table2, systematicity_df)
    summary_path = OUTPUT_DIR_DATA / "descriptive_stats_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    logger.info(f"  - descriptive_stats_summary.txt")

    # Print summary
    logger.info("\n[7/7] Phase 2 complete!")

    print("\n" + "=" * 50)
    print("PHASE 2 SUMMARY")
    print("=" * 50)
    print(f"Analytical Sample:    {len(df):,} decisions")
    print(f"Countries:            {df['a1_country_code'].nunique()}")
    print(f"Authorities:          {df['a2_authority_name'].nunique()}")
    print(f"Year Range:           {int(df['decision_year'].min())}-{int(df['decision_year'].max())}")
    print()
    print("Outputs:")
    print(f"  Tables:  {OUTPUT_DIR_TABLES}")
    print(f"  Figures: {OUTPUT_DIR_FIGURES}")
    print(f"  Data:    {OUTPUT_DIR_DATA}")
    print("=" * 50)

    # Print key findings
    print("\nKEY PRELIMINARY FINDINGS:")
    print("-" * 50)

    fine_col = "fine_amount_eur_real_2025" if "fine_amount_eur_real_2025" in df.columns else "fine_amount_eur"
    print(f"1. Median fine: €{df[fine_col].median():,.0f} (mean €{df[fine_col].mean():,.0f})")

    if "art83_discussed_count" in df.columns:
        print(f"2. Mean factors discussed: {df['art83_discussed_count'].mean():.1f} of 11")

    # Top discussed factor
    if len(table2) > 0:
        top_factor = table2.iloc[0]
        print(f"3. Most discussed factor: {top_factor['factor_display']} ({top_factor['pct_discussed']:.1f}%)")

    # Italy dominance
    italy_pct = (df['a1_country_code'] == 'IT').mean() * 100
    print(f"4. Italy dominance: {italy_pct:.1f}% of sample")

    print("-" * 50)


if __name__ == "__main__":
    main()
