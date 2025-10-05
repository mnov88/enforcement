# Phase 4 Improvement Plan

This memo captures proposed follow-up actions after reviewing the Phase 4 enrichment implementation and its dependencies.
Prioritise items according to research deadlines and data availability.

## High Priority

1. **Automated regression checks**
   - Add targeted unit tests around `compute_art5_and_rights` and `compute_measures` to ensure boolean roll-ups and QA flags
     remain stable as additional sentinel values appear.
   - Include a golden-record snapshot test for `build_graph_exports` covering node ID formatting and relationship de-duplication.

2. **FX metadata transparency**
   - Persist a supplementary table that records FX lookup method (`MONTHLY_AVG`, `ANNUAL_AVG`, `FALLBACK`) alongside the
     source year/month so analysts can filter out fallback conversions when required.
   - Consider exposing a warning list (log or CSV) for rows lacking FX coverage to streamline manual review.

## Medium Priority

1. **Context taxonomy management**
   - Externalise `CONTEXT_PRIORITY` and other enumerations to a YAML/CSV reference so non-developers can adjust ordering and
     naming without code edits.
   - Add defensive handling for new context tokens to avoid silent drops when AI annotations introduce unseen labels.
   - *Status: Completed — taxonomy now lives in `raw_data/reference/context_taxonomy.csv` with runtime loading and unmapped-token QA columns.*

2. **Article parsing enhancements**
   - Extend `parse_articles` to capture paragraph/subparagraph information (e.g., Art. 5(1)(a)) as structured columns for finer
     network analysis.
   - Backfill graph exports with edge attributes indicating article paragraph when available.
   - *Status: Completed — enrichment records `article_reference_detail`/`article_detail_tokens`, surfaces `flag_article_detail_truncated`, and propagates detail columns into graph edges.*

## Low Priority

1. **Documentation artefacts**
   - Mirror the enrichment reference (`docs/phase4_enrichment_reference.md`) into the analyst knowledge base so downstream
     consumers have a single source of truth.
   - Add ER-style diagrams illustrating how long tables relate to the enriched master dataset and graph nodes.

2. **CLI ergonomics**
   - Introduce a `--skip-graph` flag for quicker local iterations where the graph bundle is not required.
   - Emit timing metrics per major step to help diagnose slowdowns as the dataset grows.

## Recently Addressed

- Resolved the rights/profile QA bug by ensuring boolean checks rely on the `_bool` helper columns instead of raw string
  values, preventing false-positive `flag_articles_vs_rights_gap` alerts.
- Normalised measure counting so `measure_count` always reflects the integer number of sanctions applied.

Document owner: data engineering team.
