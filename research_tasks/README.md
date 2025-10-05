# Research Tasks Overview

This folder contains Python modules and documentation supporting the multi-phase
research agenda for GDPR enforcement analytics. The tasks build on the enriched
master dataset (`outputs/phase4_enrichment/1_enriched_master.csv`) and follow
the plan provided in `/research-tasks` (see project brief).

## Implemented Phases

| Task | Script | Description | Key Outputs |
| ---- | ------ | ----------- | ----------- |
| Task 0 | `scripts/rt0_sanity_check.py` | Loads the enriched dataset, enforces analysis dtypes, profiles missingness, and emits a slim `analysis_view.parquet` for downstream phases. | `outputs/research_tasks/task0/analysis_view.parquet`, `data_check.json`, readiness one-pager, missingness heatmap |
| Task 1 | `scripts/rt1_sanctions_architecture.py` | Produces sanctions incidence and mix descriptives with bootstrap confidence intervals, sanction mix index, trigger/OSS deltas, and measure co-occurrence diagnostics. | Stratified CSVs, figure bundle, `t1_summary.parquet` |
| Task 2 | `scripts/rt2_two_part_models.py` | Estimates the two-part sanction model: logistic fining incidence with IPW robustness, multinomial sanction bundle choice, OLS/quantile log-fine regressions, and scenario predictions. | Model coefficient tables, design matrix, scenario predictions, figure bundle, serialized models |
| Task 3 | `scripts/rt3_harmonization_tests.py` | Tests harmonization via nearest-neighbour fine gaps, mixed-effects variance decomposition, interaction contrasts, and within-authority public/private comparisons. | NN pair tables, variance components CSV, interaction diagnostics, figure bundle, serialized models |
| Task 4 | `scripts/rt4_factor_use_and_pack.py` | Measures Article 83(2) factor systematicity by authority, links it to dispersion metrics, and assembles publication-ready figures/tables. | Authority factor tables, systematicity index, dispersion regression outputs, figure bundle, executive summary, reproducibility README |
| Task 5 | `scripts/rt5_0_measurement_audit.py` et al. | Stress-tests the systematicity index, links it to dispersion through FE/DML/event-study designs, decomposes mechanisms, and produces forecasting scorecards. | Weighting grid, latent index draws, FE/DML tables, interaction & mediation diagnostics, policy frontier & benchmark packs |

The helper package `research_tasks` exposes reusable utilities in
`common.py` and task-specific modules (`task0.py`, `task1.py`) so future
phases can import shared loaders, constants, and writers while maintaining the
single-source data spine.

## Running Tasks

```bash
# Task 0 only
python scripts/rt0_sanity_check.py

# Task 1 only
python scripts/rt1_sanctions_architecture.py

# Task 0 → Task 4 sequentially
python run_research_tasks.py
```

Both CLI wrappers accept `--data-path` and `--output-dir` overrides. The runner
also provides `--tasks` and `--output-root` options for custom workflows.

## Outputs & Reproducibility

All artefacts are written beneath `outputs/research_tasks/<task>/` (git-ignored
in this PR). Each task records:

- `summary.txt` (≤10 lines) and `memo.txt` (5–10 lines)
- Figure pairs in PNG/PDF format
- Session metadata (`session_info.txt`) including package versions
- Binary deliverables (`.parquet`) for downstream phases

## Next Steps

- Extend Task 4 regression diagnostics with hierarchical Bayesian variants for uncertainty propagation.
- Translate systematicity dashboards into interactive notebooks for policy workshops.
- Reconcile Task 3 nearest-neighbour parameters with Task 4 dispersion controls for unified benchmarking.
