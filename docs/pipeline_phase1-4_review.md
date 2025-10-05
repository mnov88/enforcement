# Phase 1-4 Pipeline Assessment

## 1. Methodology Review (Data Science Perspective)

### Phase 1 – Parsing (`scripts/1_parse_ai_responses.py`)
**Strengths**
- Deterministic regex gate ensures blocks follow the documented delimiter structure before extracting answers.【F:scripts/1_parse_ai_responses.py†L19-L88】
- Enforces full 77-answer coverage and emits a structured error ledger for malformed cases, preserving traceability across outputs/logs.【F:scripts/1_parse_ai_responses.py†L90-L158】

**Risks / Gaps**
- Duplicate `ID` values are neither detected nor deduplicated, risking downstream overwrites when concatenated with external sources.
- `Answer N` extraction treats everything after the colon as a single-line value; multi-line answers (e.g., summaries) collapse to their first line and silently drop trailing content.
- Malformed capture only stores `str(answers)`, losing the original free-form block needed for forensic review.

### Phase 2 – Validation (`scripts/2_validate_dataset.py`)
**Strengths**
- Comprehensive schema definition covering enumerations, numeric ranges, sentinel logic, and list validation for all 77 columns.【F:scripts/2_validate_dataset.py†L18-L188】
- Layered consistency checks catch Article 5/rights misalignments and monetary inconsistencies, escalating to hard errors where violations have analytical consequences.【F:scripts/2_validate_dataset.py†L200-L276】
- Conditional rules align with documented business logic (e.g., OSS scope, fine sentinels), preventing contradictory sentinel usage.【F:scripts/2_validate_dataset.py†L278-L344】

**Risks / Gaps**
- Integer validators rely on string digit checks, so they cannot accommodate signed sentinel placeholders (e.g., `-1`) if the schema ever evolves to permit them.
- Free text validation does not cap length or detect obviously malformed placeholders (e.g., `lorem ipsum` sentinel proxies) that can signal prompt bleed-through.
- Warning vs error thresholds are static; enumerated warnings (e.g., year outside range) lack contextual tuning (no tolerance for future-dated draft decisions).

### Phase 3 – Repair (`scripts/3_repair_data_errors.py`)
**Strengths**
- Repair rules are auditable, deterministic mappings triggered only when a Phase 2 error exists, limiting accidental rewrites.【F:scripts/3_repair_data_errors.py†L1-L128】【F:scripts/3_repair_data_errors.py†L170-L233】
- Produces granular logs with per-pattern counts and sample action listings to support QA and chain-of-custody requirements.【F:scripts/3_repair_data_errors.py†L233-L321】

**Risks / Gaps**
- Rules enforce `NOT_DISCUSSED → NO` for binary fields regardless of contextual plausibility, potentially laundering genuine “unknown” states into negative assertions.
- Detailed action logs truncate after ~100 rows, so auditors must rerun the repair script or inspect CSV diffs to review the full set of automated changes.
- Manual flags (e.g., sector bleed into `a8_defendant_class`) are only counted; there is no downstream routing to ensure human follow-up.

### Phase 4 – Enrichment (`scripts/4_enrich_prepare_outputs.py`)
**Strengths**
- Comprehensive feature engineering spans temporal inference, FX harmonization, Article parsing, sanction flags, and context expansion, matching downstream analytical needs.【F:scripts/4_enrich_prepare_outputs.py†L1-L204】【F:scripts/4_enrich_prepare_outputs.py†L204-L360】
- Monetary conversions capture FX provenance (method, year, month) enabling audit trails for currency normalization.【F:scripts/4_enrich_prepare_outputs.py†L260-L360】
- Uses sentinel-aware cleaners before casting, reducing the risk of propagating placeholder strings into numeric outputs.【F:scripts/4_enrich_prepare_outputs.py†L46-L123】

**Risks / Gaps**
- FX lookup still falls back to the most recent historical rate when month/year matches are absent; use `flag_fine_fx_fallback`/`flag_turnover_fx_fallback` to exclude these rows for precision-critical analysis.【F:scripts/4_enrich_prepare_outputs.py†L220-L360】
- Reference CSVs (`context_taxonomy.csv`, `region_map.csv`) now drive context/geography expansion; missing governance around these files can trigger runtime errors or unmapped-token flags when schema changes outpace the lookup tables.【F:scripts/4_enrich_prepare_outputs.py†L24-L123】
- Enrichment emits every long table and graph bundle on each run; without CLI pruning, iterative debugging produces large diffs and heavier local processing.

## 2. Contract Compliance & Data Risk Audit (Engineering Perspective)

- **Read-only Raw Data**: All scripts respect the contract by reading from `/raw_data` and writing under `outputs/`, consistent with the repository mandate.【F:scripts/1_parse_ai_responses.py†L100-L152】【F:scripts/4_enrich_prepare_outputs.py†L1-L360】
- **Schema Alignment**: Validation rules encode current enumerations but lack automated reconciliation against `schema/` documents; drift must be manually spotted, increasing breach risk if the schema evolves without code updates.
- **Error Handling**: Absence of exception guards around file IO/CSV parsing means contract-specified runbooks may terminate abruptly on encoding anomalies, lacking graceful degradation.
- **Data Quality Threats**:
  - Potential duplicate IDs and cross-machine reuse of validation ledgers still threaten referential integrity; timestamp guards mitigate most stale-ledger cases but rely on filesystem clocks.
  - Automatic conversion of uncertain sentinel states to hard negatives (Phase 3) may violate data minimization expectations if downstream consumers assume high-confidence labels.
  - FX fallbacks are surfaced via `flag_fine_fx_fallback`/`flag_turnover_fx_fallback`, but downstream consumers must actively filter on these flags to avoid stale-rate analysis.

## 3. Improvement Plan (Highest Priority First)

1. **Critical – Strengthen Data Integrity Guards**
   - Deduplicate and collision-check `id` values during Phase 1; halt or quarantine duplicates to preserve referential integrity.【F:scripts/1_parse_ai_responses.py†L90-L158】
   - Embed schema version hashes into validation/repair outputs and verify Phase 3 consumes a matching schema snapshot before applying repairs, preventing misaligned rewrites.【F:scripts/2_validate_dataset.py†L368-L464】【F:scripts/3_repair_data_errors.py†L170-L286】

2. **High – Preserve Uncertainty & Traceability**
   - Reclassify `NOT_DISCUSSED` repairs in binary fields as warnings requiring manual adjudication or add a confidence column documenting automated coercions.【F:scripts/3_repair_data_errors.py†L24-L87】
   - Store full malformed response blocks (not just parsed dicts) in Phase 1 error output to aid forensic backtracking and contractual audit trails.【F:scripts/1_parse_ai_responses.py†L90-L158】

3. **High – Enhance Validation Robustness**
   - Replace `str.isdigit()` checks with tolerant numeric parsers that still reject empty strings and flag negative placeholders explicitly.【F:scripts/2_validate_dataset.py†L188-L248】
   - Introduce length/character filters for free-text fields and configurable warning thresholds (e.g., future-dated decisions) to reduce silent schema drift.【F:scripts/2_validate_dataset.py†L18-L188】

4. **Medium – Improve Enrichment Transparency** *(Completed)*
   - Emit explicit QA flags when FX lookup relies on fallback rates or when article parsing drops sub-article references, supporting contract-grade provenance.【F:scripts/4_enrich_prepare_outputs.py†L204-L360】
   - Externalize region/context taxonomies into reference CSVs under `raw_data/reference/` to ease updates and maintain schema compliance without code edits.【F:scripts/4_enrich_prepare_outputs.py†L24-L91】
   - ✅ Implemented via `flag_fine_fx_fallback` / `flag_turnover_fx_fallback`, `flag_article_detail_truncated`, and new reference tables (`context_taxonomy.csv`, `region_map.csv`) loaded at runtime.

5. **Medium – Operational Safeguards** *(Completed)*
   - Add CLI options and sanity checks to prevent reusing stale validation error files (e.g., compare timestamps, row counts).【F:scripts/3_repair_data_errors.py†L170-L233】
   - Wrap major IO operations in structured error handling with actionable messages to meet service-level expectations during pipeline execution.【F:scripts/run_all_pipeline.py†L1-L133】
   - ✅ Implemented `--allow-stale-errors` guard rails with timestamp/row-count checks in Phase 3 and centralized error guards in the orchestration script.

6. **Low – Analytical Enhancements**
   - Capture multi-line answers in Phase 1 using delimiter-aware parsing (e.g., storing subsequent lines until the next `Answer N` header).【F:scripts/1_parse_ai_responses.py†L66-L108】

---
Prepared by: _AI Assistant (Senior Data Scientist & Senior Developer Review)_
