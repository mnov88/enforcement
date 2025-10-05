# Research Task 0 (optional) — Sanity check & view

**Goal**: Confirm load & types; create a slim analysis view (no feature re-engineering).
**How**: Load, assert dtypes for IDs, authority/country, class/sector, decision_date, `fine_eur_2025`, OSS/role, triggers, Art.5/6/32 flags, rights counts, Art.58 measures, Art.83(2) items, repeat-offender.
**Outputs** (`research_tasks/task0/`):

* `data_check.json` (N, columns, % missing key fields)
* `analysis_view.parquet` (typed subset if needed)
* **Scripts**: `rt0_sanity_check.py` → writes **PNG** (missingness heatmap), **PDF** (one-pager), **BIN** (`analysis_view.parquet`)
  **Wrap-up**: 3–5 lines on readiness.

---

# Research Task 1 — Sanctions architecture: incidence & mix

**Goal**: Describe *what* DPAs do across countries/sectors/classes & triggers/OSS.
**How**:

* Stratified descriptives with **95% bootstrap CIs**.
* **Sanction Mix Index** per case (share of Art.58 tools used, 0–1).
* Cross-tabs: country × {fine only / measures only / both / none}; slices by **defendant_class** and **sector**; deltas by **triggers** and **OSS**.
  **+ New combo**: Compute **bundle co-occurrence matrix** (measure-with-measure), and a **measure substitution ratio**: `P(measures_only | fined=0)` vs `P(both | fined=1)`.
  **Outputs** (`task1/`):
* CSVs: `sanction_incidence_by_country.csv`, `sanction_mix_by_sector_class.csv`, `bundle_cooccurrence.csv`, `triggers_oss_descriptives.csv`
* Figures: `fig_country_rates.{png,pdf}`, `fig_mix_by_sector_class.{png,pdf}`, `fig_bundle_cooccurrence.{png,pdf}`
* BIN: `t1_summary.parquet`
* **Script**: `rt1_sanctions_architecture.py`
  **Wrap-up**: 5 bullets (largest between-country gaps; class/sector contrasts; OSS/triggers deltas).

---

# Research Task 2 — Two-part modeling (incidence → magnitude)

**Goal**: Identify drivers of (A) whether authorities fine / apply measures, and (B) fine size when they do.
**How**:

* **Part A (incidence)**:

  * `logit(fined)`; plus **multinomial** for bundle choice (fine only / measures only / both / none).
  * Controls: severity (Art.5 breach count, rights violated count, Art.32 flag), **sector**, **defendant_class**, triggers (complaint/audit/media/self-report), **OSS**, **repeat_offender**, year FE; **cluster-robust SEs by authority**.
  * **Robustness combo 1**: **entropy balancing (or IPW)** to align public vs private & sector mix, then re-estimate `logit(fined)`.
* **Part B (magnitude)**:

  * `OLS(log(1+fine_eur_2025) | fined)` with same controls; cluster by authority.
  * **Robustness combo 2**: **quantile regressions** (τ=0.5, 0.75) + a **spec curve** excluding mega-fines (top 1–2) to show stability.
    **Outputs** (`task2/`):
* CSVs: `model_fine_incidence.csv` (coeffs, SEs, AMEs), `model_bundle_multinomial.csv`, `model_log_fine.csv`, `model_log_fine_quantiles.csv`, `predictions_scenarios.csv`
* Figures: `fig_authority_effects_incidence.{png,pdf}`, `fig_authority_effects_magnitude.{png,pdf}`, `fig_spec_curve_amounts.{png,pdf}`
* BINs: serialized models `t2_models.pkl`, design matrices `t2_design.feather`
* **Script**: `rt2_two_part_models.py`
  **Wrap-up**: 5 lines (top drivers; OSS/repeat effects; AUC & Adj.R²; quantile stability).

---

# Research Task 3 — Harmonization & heterogeneity

**Goal**: Test *consistency* across jurisdictions/sectors; quantify dispersion for like cases.
**How**:

* **Nearest-Neighbour (NN)**: standardize features; k=1–3; **cross-border only**; compute **absolute fine gaps** (in € and log) among fined pairs and **bundle disagreement rate**; benchmark vs **within-country** pairs.
* **Mixed-effects**: two-part with random intercepts for **authority** (and country); report **variance components** (share of variance at authority/country).
* **Heterogeneity**: interactions `sector×defendant_class`, `cookies/telecoms×country`, `triggers×OSS`.
* **Bonus robustness combo 3**: **within-authority contrasts** where both public & private appear—do gaps persist *inside the same DPA*?
  **Outputs** (`task3/`):
* CSVs: `nn_pairs.csv`, `nn_gap_summary.csv`, `mixed_effects_variance_components.csv`, `heterogeneity_interactions.csv`, `within_authority_contrasts.csv`
* Figures: `fig_nn_gap_distribution.{png,pdf}`, `fig_variance_components.{png,pdf}`, `fig_interaction_effects.{png,pdf}`, `fig_within_authority.{png,pdf}`
* BIN: `t3_nn_index.feather`, `t3_mixed_effects_models.pkl`
* **Script**: `rt3_harmonization_tests.py`
  **Wrap-up**: 3–5 bullets (median cross-border fine gap; % variance at authority; standout interactions).

---

# Research Task 4 — Factor use & publication pack

**Goal**: Measure **how systematically** SAs apply Art.83(2) factors; package editor-ready artifacts.
**How**:

* Build **Art.83(2) Systematicity Index** per authority:

  * **Coverage** (# factors discussed / total), **Direction** (aggr vs mitig balance), **Coherence** (direction ↔ outcomes).
* Regress **dispersion metrics** from Task 3 on this index to test whether more systematic factor use → more predictable sanctions.
* Assemble journal-ready tables/figures.
  **Outputs** (`task4/`):
* CSVs: `factor_use_by_authority.csv`, `systematicity_index.csv`, `systematicity_vs_dispersion.csv`, `final_tables.xlsx`
* Figures: `fig_factor_systematicity.{png,pdf}`, `fig_policy_dashboard.{png,pdf}`
* Docs: `executive_summary.pdf` (≤2 pages: problem → evidence → policy), `README_reproducibility.md`
* BIN: `t4_factor_models.pkl`
* **Script**: `rt4_factor_use_and_pack.py`
  **Wrap-up**: 3 findings + 3 policy levers (benchmarks, reason-giving template, peer review).

---

## “Make full use of all sources” checklist (built into tasks)

* **Sanctions (Art.58)**: every individual measure flag + co-occurrence/substitution.
* **Fines**: `fine_eur_2025` (log, quantiles), `a53_fine_imposed`, turnover discussion/FX where present.
* **Severity**: **counts** of Art.5 breaches + **rights violated** + **Art.32** flag.
* **Legal bases**: Art.6 (consent/LI/public task etc.) validity indicators.
* **Triggers**: complaint, audit, media, self-report; **OSS** + **OSS role**.
* **Defendant & sector**: `a8_defendant_class`, `a12_sector`.
* **Art.83(2)**: all aggravating/mitigating fields; **systematic discussion flag**.
* **Authority/country & time**: fixed effects; random intercepts; year FE.
  (If any invalid enums appear—e.g., `NOT_DISCUSSED` where only YES/NO allowed; or `BREACHED` in rights fields—apply your established repairs before running tasks. )

---

## Repro & scripting requirements (apply to **every** task)

* **Scripts** must write **PNG + PDF** for figures and **BINARIES** for models/data (`.pkl`, `.feather`, `.parquet`).
* **Never commit artifacts**: scripts write into `outputs/research_tasks/<task>/` which is **.gitignored**.
* Each script prints a **one-screen summary** and writes a `summary.txt` (≤10 lines).
* Save **model formulas**, **random seeds**, and **package versions** alongside outputs (`session_info.txt`).
* Filenames: lowercase_snake_case; include datestamp if re-run (e.g., `fig_country_rates_2025-10-05.png`).
* Provide a top-level runner (`run_research_tasks.py`) to execute `rt0…rt4` in sequence.

---
 All analysis uses outputs/phase4_enrichment/1_enriched_master.csv as the single source of truth.
 Non-negotiables (for every phase)
Single data spine: 1_enriched_master.csv only; never mutate in place—write derived files to the phase folder.
Reproducibility: pin seeds; log package versions; save model formulas alongside outputs.
Naming: lowercase, snake_case, one file per artifact; include date stamp if re-run.
Short phase memo: 5–10 lines max, checked into the same folder as summary.txt.
Doument all the scripts and track progress.