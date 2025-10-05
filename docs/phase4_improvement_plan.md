# Phase 4 Improvement Plan

This memo captures proposed follow-up actions after reviewing the Phase 4 enrichment implementation and its dependencies. Prioritise items according to research deadlines and data availability.

## Completed Improvements

- Added regression-focused pytest coverage for Art. 5/right roll-ups, sanction aggregations, and graph exports, preventing regressions in core enrichment helpers.
- Introduced FX transparency artefacts (`0_fx_conversion_metadata.csv`, `0_fx_missing_review.csv`) so analysts can audit conversion methods and unresolved lookups.
- Externalised the processing-context taxonomy and region groupings into reference CSVs loaded at runtime, with QA flags for unmapped tokens.
- Extended article parsing to retain paragraph/subparagraph detail (`article_reference_detail`, `article_detail_tokens`) and surface truncation flags in both tabular and graph outputs.

## Remaining Opportunities

### Low Priority

1. **Documentation artefacts**
   - Mirror the enrichment reference (`docs/phase4_enrichment_reference.md`) into the analyst knowledge base so downstream consumers have a single source of truth.
   - Add ER-style diagrams illustrating how long tables relate to the enriched master dataset and graph nodes.

2. **CLI ergonomics**
   - Introduce a `--skip-graph` flag for quicker local iterations where the graph bundle is not required.
   - Emit timing metrics per major step to help diagnose slowdowns as the dataset grows.

## Recently Addressed

- Resolved the rights/profile QA bug by ensuring boolean checks rely on the `_bool` helper columns instead of raw string values, preventing false-positive `flag_articles_vs_rights_gap` alerts.
- Normalised measure counting so `measure_count` always reflects the integer number of sanctions applied.

Document owner: data engineering team.
