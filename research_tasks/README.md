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

# Task 0 → Task 1 sequentially
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

- Task 2: two-part sanction modelling (incidence & magnitude)
- Task 3: harmonisation and heterogeneity analysis
- Task 4: Article 83(2) factor systematicity and publication pack

These modules will hook into the same loader and output conventions introduced
in Tasks 0 and 1.
