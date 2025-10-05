# Phase 4 Enrichment Reference

This document provides granular documentation for `scripts/4_enrich_prepare_outputs.py` and the artefacts written to
`outputs/phase4_enrichment/`. Review it alongside `readme.md` and `SCRIPTS-README.md` when maintaining the enrichment
stage or onboarding new collaborators.

## Inputs and Supporting Tables

| Source | Description |
| --- | --- |
| `outputs/phase3_repair/repaired_dataset.csv` | Primary Phase 3 output. One row per enforcement decision with 77 schema columns. |
| `raw_data/reference/fx_rates.csv` | Monthly and annual FX factors used to convert fines and turnover into EUR. |
| `raw_data/reference/hicp_ea19.csv` | Euro area HICP index (2016–2025) for expressing monetary values in 2025 EUR. |
| `raw_data/reference/context_taxonomy.csv` | Processing-context taxonomy with priority ordering and canonical flag column names. |
| `raw_data/reference/region_map.csv` | Analyst-defined region groupings for EU/EEA country codes. |

The enrichment script consumes the repaired dataset, joins FX and inflation references in-memory, and emits derived tables.

## Execution Flow

1. **Temporal scaffolding**
   - `compute_temporal_features` normalises `a4_decision_year` and `a5_decision_month`, derives quarter numbers, and
     constructs an inferred `decision_date_inferred` timestamp for rows with both year and month.
   - `temporal_granularity` flags whether a row captures full month information (`YEAR_MONTH`) or only the year.

2. **Currency normalisation**
   - `convert_monetary_columns` standardises fines (`a54`, `a55`) and turnover (`a57`, `a58`) to EUR using the lookup rules below.
   - Lookup precedence: exact month → annual average → most recent fallback rate.
   - Additional outputs include `fine_fx_method`, FX source year/month, `fine_amount_log`, bucketed ranges, and 2025 EUR
     deflations (`fine_amount_eur_real_2025`, `turnover_amount_eur_real_2025`).
   - Flags: `fine_present`, `turnover_present`, and `fine_pct_turnover` (omitted for defendants without turnover data).
   - QA markers: `flag_fine_fx_fallback` and `flag_turnover_fx_fallback` highlight rows that fall back to out-of-period FX rates.

3. **Article 5 & rights indicators**
   - Binary booleans are emitted for every Art. 5 answer (e.g., `art5_1_a_breached_bool`).
   - Rights violations (`a37`–`a44`) are mapped to booleans, with `rights_violated_count` and a compact
     `rights_profile` string that concatenates GDPR article codes.
   - `breach_count_total`, `breach_has_artX` flags, and `breach_family_top` summarise articles parsed from
    `a77_articles_breached` (via `parse_articles`).
   - `flag_article_detail_truncated` captures decisions where citations include paragraph/sub-paragraph detail (e.g.,
     `Art. 5(1)(a)`) that are reduced to the parent article number; the long table retains `article_reference_detail`
     and `article_detail_tokens` for provenance.
   - QA checks: `flag_articles_vs_rights_gap` (rights claimed without matching Article 77 citations) and
     `flag_art5_breached_but_5_not_in_77`.

4. **Sanction measures & Article 83 scoring**
   - `compute_measures` exposes booleans for each sanction type (`a45`–`a52`), counts (`measure_count`), and roll-ups such as
     `sanction_profile`, `is_warning_only`, and `is_fine_only`.
   - `compute_art83_scores` converts qualitative factors (`a59`–`a69`) into numeric scores, totals aggravating vs mitigating
     factors, and captures systematic discussion metadata (`art83_systematic_bool`, `first_violation_status`).

5. **Context and complaint features**
   - `explode_semicolon_list` expands semicolon-delimited columns into long tables (`2_processing_contexts.csv`,
     `3_vulnerable_groups.csv`, `4_guidelines.csv`).
   - `compute_context_features` now loads `context_taxonomy.csv`, producing boolean columns for every configured
     processing context along with context counts and keys such as `sector_x_context_key` and `role_sector_key`.
   - QA columns `flag_context_token_unmapped` and `context_unknown_tokens` surface annotations that are not yet defined in the taxonomy file.
   - `compute_complaint_and_flags` tracks complaints, audits, Article 33 discussions, and exposes
     `flag_art33_inconsistency` where breach notifications are evaluated despite no OSS discussion.

6. **OSS, geography, and text metadata**
   - `compute_oss_and_geography` classifies OSS posture (`oss_case_category`, `oss_role_lead_bool`, `oss_role_concerned_bool`)
    and harmonises authority names while attaching regional groupings sourced from `region_map.csv`.
   - `compute_text_features` records text lengths for narrative fields (`*_len` suffixes), mines keyword tags
     (`keywords_legal_basis`), and counts cited guidelines.

7. **Quality signals**
   - `compute_quality_flags` introduces QA markers for missing sector detail, absent currencies where fines exist, and
     mismatches between systematic Article 83 discussions and individual factor coverage.

8. **Graph exports**
   - `build_graph_exports` creates Neo4j-ready node and edge CSVs covering decisions, authorities, defendants, articles,
     guidelines, and processing contexts. Node IDs follow the `Type|SLUG` convention used by the analyst team.

## Output Catalogue

| File | Granularity | Highlights |
| --- | --- | --- |
| `1_enriched_master.csv` | Decision-level | All engineered features described above plus passthrough schema fields. |
| `2_processing_contexts.csv` | Long table | One row per decision-context pair, retaining the token order in `a14`. |
| `3_vulnerable_groups.csv` | Long table | Exploded vulnerable group annotations from `a29`. |
| `4_guidelines.csv` | Long table | Guidelines from `a74`, with positional ordering. |
| `5_articles_breached.csv` | Long table | Parsed article references, numeric identifiers, and linkable labels (e.g., `GDPR_5`). |
| `graph/nodes_*.csv`, `graph/edges_*.csv` | Graph | Bulk import extracts for Neo4j (Decisions ↔ Authorities/Defendants/Articles/Guidelines/Contexts). |

### Column Naming Notes

- Boolean helper columns follow the `<source>_bool` or `<concept>_flag` conventions for consistent filtering.
- Monetary metrics use suffixes `_eur`, `_eur_real_2025`, `_bucket`, or `_log` to show aggregation intent.
- QA outputs (`flag_*`) always evaluate to Python booleans, enabling direct filtering without coercion.

## Re-running Phase 4

```bash
python scripts/4_enrich_prepare_outputs.py \
  --input outputs/phase3_repair/repaired_dataset.csv \
  --output outputs/phase4_enrichment \
  --fx-table raw_data/reference/fx_rates.csv \
  --hicp-table raw_data/reference/hicp_ea19.csv \
  --context-taxonomy raw_data/reference/context_taxonomy.csv \
  --region-map raw_data/reference/region_map.csv
```

All outputs are overwritten on each run. Commit regenerated CSVs only when the upstream data or enrichment logic changes.
