# Data Pipeline Scripts Documentation

This document provides comprehensive documentation for all five phases of the GDPR enforcement data pipeline.

---

## Phase 1: Data Extraction

### Script: `scripts/1_parse_ai_responses.py`

**Purpose:** Parse AI-generated responses from delimited text format into structured CSV.

**Input:**
- `/raw_data/AI_analysis/AI-responses.txt` (1,518 responses in delimited text format)

**Output:**
- `/outputs/phase1_extraction/main_dataset.csv` (1,473 rows × 77 fields)
- `/outputs/phase1_extraction/extraction_log.txt` (parsing statistics + malformed roster)
- `/outputs/phase1_extraction/data_with_errors.csv` (45 malformed responses for audit)

**Data Transformations:**

1. **Text Parsing:**
   - Extracts responses between `----- RESPONSE DELIMITER -----` markers
   - Parses ID from header line: `ID: [case_identifier]`
   - Extracts 77 answer lines following format: `Answer N: [value]`

2. **Field Mapping:**
   - `Answer 1` → `a1_country_code`
   - `Answer 2` → `a2_authority_name`
   - `Answer 3` → `a3_appellate_decision`
   - ... (77 total fields)
   - `Answer 77` → `a77_articles_breached`

3. **Data Quality:**
   - Validates each response has exactly 77 answer lines
   - Skips incomplete responses (e.g., missing answers)
   - Trims whitespace from all values
   - Preserves empty answers as empty strings

4. **Results:**
   - Successfully parsed: 1,473/1,518 responses (97.0%)
   - Malformed/skipped: 45 responses (3.0%, typically truncated after Q73 or empty bodies)

**Key Fix:** Original regex pattern required double delimiter before ID line, causing 50% data loss. Fixed to single delimiter pattern.

**Usage:**
```bash
python3 scripts/1_parse_ai_responses.py
```

---

## Phase 2: Data Validation

### Script: `scripts/2_validate_dataset.py`

**Purpose:** Validate all 77 fields against schema rules and identify errors/warnings.

**Input:**
- `/outputs/phase1_extraction/main_dataset.csv` (default)
- Or custom path via `--input` flag

**Output:**
- `/outputs/phase2_validation/validated_data.csv` (rows with no errors)
- `/outputs/phase2_validation/validation_errors.csv` (detailed error report)
- `/outputs/phase2_validation/validation_report.txt` (summary statistics)

_Latest run (2025-10-05): 742 rows passed validation, 731 rows with issues, and 2,800 total problems logged (2,501 errors / 299 warnings)._

**Validation Rules (8 Types):**

1. **`enum`** – Exact match against the schema list (e.g., `a1_country_code`).
2. **`integer_range`** – Bounded integer checks with optional warning ranges (e.g., decision year/month).
3. **`free_text`** – Non-empty text enforcement for required narrative fields (e.g., authority name, case summary).
4. **`free_text_or_sentinel`** – Text allowed when populated, otherwise a single sentinel (e.g., `a13_sector_other`).
5. **`integer_or_sentinel`** – Accepts positive integers or a sentinel token (e.g., `a54_fine_amount`).
6. **`integer_or_sentinels`** – Accepts positive integers or multiple sentinels (e.g., `a57_turnover_amount`).
7. **`semicolon_list_or_sentinel`** – Semicolon-delimited enums or a sentinel (e.g., `a14_processing_contexts`, `a29_vulnerable_groups`).
8. **`semicolon_integers_or_sentinels`** – Semicolon-delimited article numbers with normalization of "Art." prefixes and paragraph detail (e.g., `a77_articles_breached`).

**Conditional Cross-Field Rules (8 Rules):**

1. `a10_gov_level = NOT_APPLICABLE` when `a8_defendant_class ≠ PUBLIC`
2. `a13_sector_other = NOT_APPLICABLE` when `a12_sector ≠ OTHER`
3. `a19_art33_breached = NOT_APPLICABLE` when `a18_art33_discussed = NO`
4. `a54_fine_amount = NOT_APPLICABLE` when `a53_fine_imposed = NO` (ERROR)
5. `a55_fine_currency = NOT_APPLICABLE` when `a53_fine_imposed = NO` (ERROR)
6. `a57_turnover_amount = NOT_DISCUSSED` when `a56_turnover_discussed = NO`
7. `a58_turnover_currency = NOT_DISCUSSED` when `a56_turnover_discussed = NO`
8. `a73_oss_role = NOT_APPLICABLE` when `a72_cross_border_oss ≠ YES`

**Hard Consistency Checks (phase-2 errors):**

- If `a53_fine_imposed = YES`, then `a54_fine_amount` must be a positive integer and `a55_fine_currency` must be a real ISO code (no sentinels).
- If any of `a21`–`a27` = BREACHED, then `a77_articles_breached` must include `5`.
- If any rights field (`a37`–`a44`) = YES, then `a77_articles_breached` must list the corresponding article(s) (e.g., `a43 = YES` ⇒ Article 12/13/14 present).

**No Data Modification:** This script only validates and reports errors; it does not modify data.

**Usage:**
```bash
# Validate main dataset
python3 scripts/2_validate_dataset.py

# Validate custom dataset
python3 scripts/2_validate_dataset.py --input outputs/phase3_repair/repaired_dataset.csv
```
Running with `--input` writes `<dataset>_validated.csv`,
`<dataset>_validation_errors.csv`, and `<dataset>_validation_report.txt`
next to the provided file to avoid overwriting Phase 2 outputs.

---

## Phase 2 Utility: Enum Value Analysis

### Script: `scripts/2_analyze_enum_values.py`

**Purpose:** Analyze frequency of all enum field values to identify common mismappings.

**Input:**
- `/outputs/phase1_extraction/main_dataset.csv` (default)
- Or custom path via `--input` flag

**Output:**
- `/outputs/phase2_validation/enum_analysis.txt` (human-readable report)
- `/outputs/phase2_validation/enum_analysis.csv` (machine-readable data)

**Analysis:**

1. **Frequency Counting:**
   - Counts occurrences of each value in all 62 enum fields
   - Marks values as valid (✓) or invalid (✗) based on schema

2. **Report Format (TXT):**
   ```
   a43_transparency_violated
   Schema allows: NO, NOT_DISCUSSED, YES

   Valid values:
     ✓ NOT_DISCUSSED                    <count>
     ✓ YES                              <count>
     ✓ NO                               <count>

   Invalid values (NOT IN SCHEMA):
     ✗ BREACHED                         <count>  ← FIX NEEDED
     ✗ NOT_APPLICABLE                   <count>  ← FIX NEEDED

   Summary: <invalid>/<total> values are invalid (counts update with each run)
   ```

3. **CSV Output:**
   ```
   field_name,value,count,valid,status
   a43_transparency_violated,BREACHED,47,NO,✗ FIX
   ```

**No Data Modification:** This is a read-only analysis utility.

**Maintenance tip:** When schema enums expand (e.g., new currency codes), update the `ENUM_FIELDS` map here to keep the analysis in sync with the validator.

**Usage:**
```bash
# Analyze main dataset
python3 scripts/2_analyze_enum_values.py

# Analyze custom dataset
python3 scripts/2_analyze_enum_values.py --input outputs/phase3_repair/repaired_dataset.csv
```

---

## Phase 3: Data Repair

### Script: `scripts/3_repair_data_errors.py`

**Purpose:** Automatically fix common validation errors using pattern-based repair rules.

**Input:**
- `/outputs/phase1_extraction/main_dataset.csv` (data to repair)
- `/outputs/phase2_validation/validation_errors.csv` (errors to fix)

**Output:**
- `/outputs/phase3_repair/repaired_dataset.csv` (1,473 rows with targeted repairs applied)
- `/outputs/phase3_repair/repaired_dataset_validated.csv` (941-row clean subset after revalidation)
- `/outputs/phase3_repair/repaired_dataset_validation_errors.csv`
- `/outputs/phase3_repair/repaired_dataset_validation_report.txt`
- `/outputs/phase3_repair/repair_log.txt` (detailed repair log + pattern counts)

**Repair Rules (5 Automatic Patterns + 1 Flag):**

### Pattern 1: `NOT_DISCUSSED → NO`
**Fields (5):** a18_art33_discussed, a28_art9_discussed, a53_fine_imposed, a56_turnover_discussed, a70_systematic_art83_discussion

**Reason:** Two-option fields expect YES or NO only; NOT_DISCUSSED is semantically invalid

**Example:**
- Before: `a53_fine_imposed = NOT_DISCUSSED`
- After: `a53_fine_imposed = NO`

---

### Pattern 2: `NOT_APPLICABLE → NOT_DISCUSSED`
**Fields (47):** a3_appellate_decision; a15-a17 (case origins); a20_breach_notification_effect; a21-a35 (Articles 5 & 6); a37-a44 (rights); a45-a52 (corrective measures); a59-a69 (Art 83 factors); a71_first_violation; a72_cross_border_oss

**Reason:** NOT_APPLICABLE used incorrectly on fields that accept NOT_DISCUSSED sentinel

**Example:**
- Before: `a43_transparency_violated = NOT_APPLICABLE`
- After: `a43_transparency_violated = NOT_DISCUSSED`

---

### Pattern 3: `BREACHED → YES`
**Fields (8):** a37_right_access_violated, a38_right_rectification_violated, a39_right_erasure_violated, a40_right_restriction_violated, a41_right_portability_violated, a42_right_object_violated, a43_transparency_violated, a44_automated_decisions_violated

**Reason:** Rights violation fields expect YES/NO/NOT_DISCUSSED, not BREACHED/NOT_BREACHED terminology

**Example:**
- Before: `a43_transparency_violated = BREACHED`
- After: `a43_transparency_violated = YES`

---

### Pattern 4: `NOT_BREACHED → NO`
**Fields (9):** a37-a44 (rights violated), a19_art33_breached

**Reason:** Field expects YES/NO format, not Article 5 principles terminology

**Example:**
- Before: `a39_right_erasure_violated = NOT_BREACHED`
- After: `a39_right_erasure_violated = NO`

---

### Flag 5: `a8_defendant_class` sector labels
**Field (1):** a8_defendant_class

**Values Flagged:** MEDIA, EDUCATION, JUDICIAL, TELECOM (left untouched)

**Reason:** Sector labels captured in the defendant-class field require human review; no automatic remapping is applied.

**Example:**
- Logged: `a8_defendant_class = MEDIA` → `FLAG`
- Analyst should confirm correct class (e.g., PUBLIC/PRIVATE) before publication.

---

### Pattern 6: `INTENTIONAL → AGGRAVATING`
**Field (1):** a60_intentional_negligent

**Reason:** Field expects AGGRAVATING/MITIGATING/NEUTRAL/NOT_DISCUSSED, not specific fault terminology

**Example:**
- Before: `a60_intentional_negligent = INTENTIONAL`
- After: `a60_intentional_negligent = AGGRAVATING`

> ❗ Binary fields (YES/NO only) are no longer auto-repaired when set to `NOT_APPLICABLE`. Leave these to manual review so context is not lost.

**Repair Logic:**
1. Only repairs fields with validation errors (not warnings)
2. Applies repairs only when field value matches rule condition
3. Preserves original data structure (no columns added/removed)
4. Logs every action (repair or flag) with row ID, field, old value, new value/action

**Data Modifications:**
- Changes enum values to schema-compliant alternatives
- Records flagged-but-unmodified values for manual follow-up
- Does NOT modify numeric values, free text, or dates
- Does NOT delete or add rows
- Does NOT change field names or structure

**Operational Safeguards:**
- Compares dataset and validation-ledger timestamps; aborts when the ledger is older unless `--allow-stale-errors` is supplied.
- Verifies the error ledger references only IDs present in the dataset and matches declared row counts before applying repairs.

**Usage:**
```bash
python3 scripts/3_repair_data_errors.py \
  --input outputs/phase1_extraction/main_dataset.csv \
  --errors outputs/phase2_validation/validation_errors.csv \
  --output outputs/phase3_repair/repaired_dataset.csv

# Override safeguards when intentionally reusing an older ledger
python3 scripts/3_repair_data_errors.py --allow-stale-errors
```

---

## Complete Pipeline Workflow

### Step 1: Extract
```bash
python3 scripts/1_parse_ai_responses.py
```
**Result:** 1,518 → 1,473 rows (97.0% extraction rate)

### Step 2: Validate (Initial)
```bash
python3 scripts/2_validate_dataset.py
```
**Result:** 742/1,473 valid rows (50.4%)

### Step 3: Analyze Patterns
```bash
python3 scripts/2_analyze_enum_values.py
```
**Result:** See `outputs/phase2_validation/enum_analysis.txt` for field-by-field invalid token counts (updated each run).

### Step 4: Repair
```bash
python3 scripts/3_repair_data_errors.py
```
**Result:** 1,016 automatic repairs + 18 manual flags (406 rows touched) in the latest run; review `repair_log.txt` for details.

### Step 5: Validate (Post-Repair)
```bash
python3 scripts/2_validate_dataset.py --input outputs/phase3_repair/repaired_dataset.csv
```
**Result:** 941/1,473 valid rows (63.9%) - **+199 rows improved, +13.5%**

---

## Data Quality Metrics

| Metric | Initial | After Phase 3 | Change |
|--------|---------|---------------|--------|
| **Valid Rows** | 742/1,473 (50.4%) | 941/1,473 (63.9%) | +199 rows (+13.5%) |
| **Total Errors** | 2,800 | 2,026 | -774 issues (-27.6%) |
| **Schema Violations (ERROR)** | 2,501 | 1,642 | -859 errors (-34.3%) |
| **Suspicious Values (WARNING)** | 299 | 384 | +85 warnings (+28.4%) |

---

## Schema Updates Made

During development, the following values were added to schema to reflect actual data:

1. **a1_country_code:** Added 'EL' (Greece alternative code)
2. **a8_defendant_class:** Added 'POLITICAL_PARTY'
3. **a14_processing_contexts:** Added 'EMPLOYEE_MONITORING'
4. **a55_fine_currency:** Added 'ISK' (Icelandic Króna), 'USD' (US Dollar), and 'CHF' (Swiss Franc)
5. **a58_turnover_currency:** Added 'ISK' (Icelandic Króna), 'USD' (US Dollar), and 'CHF' (Swiss Franc)

---

## File Locations Summary

### Input
- Raw data: `/raw_data/AI_analysis/AI-responses.txt`

### Phase 1 Output
- Main dataset: `/outputs/phase1_extraction/main_dataset.csv`
- Extraction log: `/outputs/phase1_extraction/extraction_log.txt`

### Phase 2 Output
- Valid rows: `/outputs/phase2_validation/validated_data.csv`
- Error details: `/outputs/phase2_validation/validation_errors.csv`
- Summary report: `/outputs/phase2_validation/validation_report.txt`
- Enum analysis (text): `/outputs/phase2_validation/enum_analysis.txt`
- Enum analysis (CSV): `/outputs/phase2_validation/enum_analysis.csv`

### Phase 3 Output
- Repaired dataset: `/outputs/phase3_repair/repaired_dataset.csv`
- Repair log: `/outputs/phase3_repair/repair_log.txt`
- Re-validated data: `/outputs/phase3_repair/repaired_dataset_validated.csv`
- Re-validation errors: `/outputs/phase3_repair/repaired_dataset_validation_errors.csv`
- Re-validation report: `/outputs/phase3_repair/repaired_dataset_validation_report.txt`

### Phase 4 Output
- Enriched master dataset: `/outputs/phase4_enrichment/1_enriched_master.csv`
- FX diagnostics: `/outputs/phase4_enrichment/0_fx_conversion_metadata.csv`
- FX missing review list: `/outputs/phase4_enrichment/0_fx_missing_review.csv`
- Long tables: `2_processing_contexts.csv`, `3_vulnerable_groups.csv`, `4_guidelines.csv`, `5_articles_breached.csv`
- Graph bundle: `/outputs/phase4_enrichment/graph/`

### Phase 5 Output
- Cohort + modelling artefacts in `/outputs/phase5_analysis/` (case-level features, cohorts, contrasts, mixed-effects results)

---

## Phase 4: Enrichment & Multi-format Delivery

### Script: `scripts/4_enrich_prepare_outputs.py`

**Purpose:** Convert the repaired Phase 3 dataset into a feature-complete analytical package and graph-friendly extracts.

**Inputs:**
- `/outputs/phase3_repair/repaired_dataset.csv`
- FX lookups: `/raw_data/reference/fx_rates.csv`
- HICP deflators: `/raw_data/reference/hicp_ea19.csv`
- Context taxonomy: `/raw_data/reference/context_taxonomy.csv`
- Region groupings: `/raw_data/reference/region_map.csv`

**Outputs:**
- `/outputs/phase4_enrichment/0_fx_conversion_metadata.csv` – Captures FX lookup method (`MONTHLY_AVG`, `ANNUAL_AVG`, `FALLBACK`), source year/month, and converted values for fines and turnover.
- `/outputs/phase4_enrichment/0_fx_missing_review.csv` – Summarises cases where FX conversion failed so reviewers can intervene.
- `/outputs/phase4_enrichment/1_enriched_master.csv` – Adds temporal fields, FX/EUR normalization (nominal & real 2025 euros), sanction and Art. 83 scoring, contextual flags, OSS geography, QA signals, and keyword metadata.
- `/outputs/phase4_enrichment/2_processing_contexts.csv` – Long table of processing contexts with positional order.
- `/outputs/phase4_enrichment/3_vulnerable_groups.csv` – Exploded vulnerable group annotations.
- `/outputs/phase4_enrichment/4_guidelines.csv` – Guidelines referenced per decision.
- `/outputs/phase4_enrichment/5_articles_breached.csv` – Parsed GDPR articles with numeric anchors and preserved detail tokens.
- `/outputs/phase4_enrichment/graph/` – Neo4j bulk-import node and edge CSVs for Decisions, Authorities, Defendants, Articles, Guidelines, and Contexts.

For a step-by-step explanation of each enrichment helper and derived column family, consult
`docs/phase4_enrichment_reference.md`.

**Key Transformations:**

1. **Temporal normalization** – Derives `decision_year`, `decision_month`, inferred dates, quarter buckets, and granularity flags even when only the year is known.
2. **Monetary harmonization** – Maps all fines/turnover to EUR using ECB-informed FX tables, adds deflated 2025 EUR values, log scaling, ratio metrics, categorical buckets, and QA flags (`flag_fine_fx_fallback`, `flag_turnover_fx_fallback`) when conversions rely on fallback rates.
3. **Rights & breaches** – Computes Art. 5 boolean flags, rights violation profiles, per-article presence flags, priority breach families, and highlights sub-article detail via `flag_article_detail_truncated` plus detail columns in long tables and graph edges.
4. **Sanctions & Article 83 scoring** – Produces sanction profiles, measure counts, warning/fine convenience booleans, and numerical Art. 83 factor scores with coverage counts.
5. **Contextual cross-features** – Loads `context_taxonomy.csv` to expand processing contexts into binary indicators, builds `sector_x_context_key`, surfaces `context_profile`, and records unmapped tokens via `flag_context_token_unmapped`/`context_unknown_tokens`.
6. **Quality and QA flags** – Highlights sector detail gaps, currency omissions, Article 33 inconsistencies, and systematic-factor mismatches.
7. **Graph exports** – Emits node/edge CSVs aligning with the schema described in the analyst brief for direct Neo4j bulk import or further network analysis.

**Usage:**
```bash
python scripts/4_enrich_prepare_outputs.py \
  --input outputs/phase3_repair/repaired_dataset.csv \
  --output outputs/phase4_enrichment \
  --fx-table raw_data/reference/fx_rates.csv \
  --hicp-table raw_data/reference/hicp_ea19.csv \
  --context-taxonomy raw_data/reference/context_taxonomy.csv \
  --region-map raw_data/reference/region_map.csv
```

---

## Phase 5: Similarity Analysis & Modelling

### Script: `scripts/5_analysis_similarity.py`

**Purpose:** Translate enriched cases into article-based similarity cohorts, run within-cohort contrasts for specific factors, match comparable cases across countries, and estimate mixed-effects regressions on log fines.

**Input:**
- `/outputs/phase4_enrichment/1_enriched_master.csv`

**Primary Outputs (`outputs/phase5_analysis/`):**

1. `0_case_level_features.csv` – Parses `a77_articles_breached` into integer sets (`article_set_key`, `article_family_key`), builds Art. 58(2) measure sets, and stores log fines (2025 EUR).
2. `1_baseline_article_cohorts.csv` – Exact article cohorts with log-fine dispersion, mean measure-set Jaccard similarity, and sanction profile modes.
3. `2_case_level_with_components.csv` & `2_relaxed_article_components.csv` – Adds relaxed cohorts via Jaccard ≥ 0.8 union-find clustering for sparse article combinations.
4. `3_context_effects.csv` – Stratified Mann–Whitney contrasts for context flags (e.g., CCTV, employee monitoring) that hold other contexts, roles, sectors, and defendant type constant.
5. `3_legal_basis_effects.csv` – Keeps `NOT_DISCUSSED` separate while comparing Art. 6 invalid vs valid/not discussed outcomes.
6. `3_defendant_type_effects.csv` – Private vs public comparisons scoped to fixed context bundles (employment, CCTV, marketing, etc.).
7. `4_cross_country_pairs.csv` & `4_cross_country_summary.csv` – Greedy nearest-neighbour matches in different countries within the same article cohort, with paired t-statistics and McNemar counts.
8. `5_mixed_effects_results.csv` & `5_mixed_effects_summary.txt` – Mixed-effects estimates for log fines with article-set random intercepts plus leave-one-layer-out variants.
9. `6_relaxed_cohort_contrasts.csv` – Sensitivity check comparing context contrasts under relaxed cohorts.
10. `6_time_controls_summary.csv` – Period buckets (pre-2021 vs 2021+) for cohort-level means of log fines, measure counts, and fine incidence.

**Usage:**
```bash
python scripts/5_analysis_similarity.py
```

> ⚠️ Statsmodels issues warnings about singular random-effect covariance matrices on sparse cohorts. The script preserves the coefficient tables but flags the limitation in `readme.md`.

---

## Research Tasks (Phase 5+ analytics)

### Script: `scripts/rt0_sanity_check.py`

**Purpose:** Implements Research Task 0 by loading the enriched master dataset, enforcing analysis-friendly dtypes, profiling missingness, and generating an `analysis_view.parquet` alongside a readiness one-pager and heatmap.

**Outputs:** `outputs/research_tasks/task0/` (analysis view, `data_check.json`, summary/memo, figures, session info).

**Usage:**
```bash
python scripts/rt0_sanity_check.py
```

### Script: `scripts/rt1_sanctions_architecture.py`

**Purpose:** Delivers Research Task 1 by estimating sanctions incidence/mix descriptives with 95% bootstrap CIs, constructing the sanction mix index, trigger/OSS deltas, and Art. 58 measure co-occurrence diagnostics.

**Outputs:** `outputs/research_tasks/task1/` (CSV tables, figure bundle, `t1_summary.parquet`, summary/memo, session info).

**Usage:**
```bash
python scripts/rt1_sanctions_architecture.py
```

### Script: `scripts/rt2_two_part_models.py`

**Purpose:** Executes Research Task 2 by estimating two-part sanction models: logistic fining incidence with cluster-robust and IPW specifications, multinomial bundle choice (fine-only / measures-only / both / neither), and log-fine magnitude regressions with OLS, quantile fits, and specification-curve robustness.

**Outputs:** `outputs/research_tasks/task2/` (model coefficient CSVs, scenario predictions, design matrix feather, serialized models, figure bundle, summary/memo, session info).

**Usage:**
```bash
python scripts/rt2_two_part_models.py
```

### Script: `scripts/rt3_harmonization_tests.py`

**Purpose:** Runs Research Task 3 nearest-neighbour dispersion diagnostics, Bayesian mixed-effects models, and interaction contrasts for harmonisation testing.

**Outputs:** `outputs/research_tasks/task3/` (NN tables, variance components, interaction figures, summary/memo, session info).

**Usage:**
```bash
python scripts/rt3_harmonization_tests.py
```

### Script: `scripts/rt4_factor_use_and_pack.py`

**Purpose:** Executes Research Task 4 to assemble systematicity indices, dispersion regressions, and publication-ready dashboards.

**Outputs:** `outputs/research_tasks/task4/` (systematicity tables, dispersion regression outputs, policy figures, summary/memo, session info).

**Usage:**
```bash
python scripts/rt4_factor_use_and_pack.py
```

### Scripts: `scripts/rt5_0_measurement_audit.py` → `scripts/rt5_4_forecasting_benchmarks.py`

**Purpose:** Implement Research Task 5 modules covering measurement robustness, index sensitivity, dispersion inference, mechanism decomposition, and forecasting/benchmarking.

**Outputs:** `outputs/research_tasks/task5/` (weighting grid, latent index draws, FE/DML diagnostics, interaction & mediation tables, policy frontier & benchmark artefacts, summary/memo, session info).

**Usage (any module invokes the full Task 5 run):**
```bash
python scripts/rt5_0_measurement_audit.py
```

### Runner: `run_research_tasks.py`

Sequential orchestrator for `rt0` → `rt5`. Accepts `--tasks`, `--data-path`, and `--output-root` overrides to support reproducible pipelines.

```bash
python run_research_tasks.py --tasks task0 task3 task5
```

All research-task artefacts live under `outputs/research_tasks/` and are git-ignored by default.

---

## Phase 6: Paper Analysis (Methodology Implementation)

This phase implements the methodology proposal for the research paper "Reasoning and Consistency in GDPR Enforcement: A Cross-Border Analysis of Penalty Factors and Fine Disparities".

### Script: `scripts/6_paper_data_preparation.py`

**Purpose:** Phase 1 of the paper analysis pipeline - data preparation, systematicity index computation, and cohort construction.

**Input:**
- `/outputs/phase4_enrichment/1_enriched_master.csv` (enriched master dataset)
- `/raw_data/reference/region_map.csv` (country-to-region mapping)

**Output:**
- `/outputs/paper/data/analysis_sample.csv` (528 rows - fine-imposed decisions)
- `/outputs/paper/data/authority_systematicity.csv` (22 authorities with ≥10 decisions)
- `/outputs/paper/data/cohort_membership.csv` (229 unique article cohorts)
- `/outputs/paper/data/sample_construction_log.txt` (sample flow documentation)

**Data Transformations:**

1. **Sample Construction:**
   - Filters for `a53_fine_imposed = YES` AND `fine_amount_eur > 0`
   - Validates country codes to exclude malformed records
   - Adds cross-border eligibility flag (article cohort in ≥2 countries)
   - Result: 528 analytical decisions (316 cross-border eligible)

2. **Systematicity Index Computation:**
   - For each authority with ≥10 decisions, computes:
     - **Coverage:** `mean(art83_discussed_count) / 11` (factor completeness)
     - **Consistency:** `1 - normalized_std(balance_score)` (directional stability)
     - **Coherence:** `|cor(balance_score, log_fine)|` (outcome alignment)
     - **Systematicity:** `Coverage × Consistency × Coherence`
   - Range: [0, 1] where higher = more systematic reasoning
   - Results: 22 authorities indexed (range: 0.00 - 0.25)

3. **Article Cohort Generation:**
   - Parses `a77_articles_breached` into sorted numeric sets
   - Creates `article_set_key` (e.g., "5;6;15")
   - Creates `article_family_key` for relaxed matching
   - Computes cohort statistics (case count, country count, mean/median fines)

**Key Variables Added:**
- `article_set_key` - Exact article set identifier
- `article_family_key` - Article family grouping for relaxed matching
- `article_count` - Number of breached articles
- `cross_border_eligible` - Boolean flag for cross-border analysis
- `log_fine_2025` - Natural log of inflation-adjusted fine (EUR, 2025)
- `region` - EU region grouping

**Usage:**
```bash
python scripts/6_paper_data_preparation.py
```

---

## Important Notes

1. **Phase 2 validation is non-destructive** - it only reads and reports, never modifies data
2. **Phase 3 repairs are deterministic** - same input always produces same output
3. **Always re-run validation after repair** to verify improvements
4. **Enum analysis is a diagnostic tool** - use it to identify new repair patterns
5. **Schema updates require changes to both** `main-schema-critically-important.md` and `AI-prompt-very-important.md`

---

## Adding New Repair Rules

To add a new repair pattern to Phase 3:

1. Run enum analysis to identify the pattern
2. Add pattern to `REPAIR_RULES` dict in `scripts/3_repair_data_errors.py`
3. Specify affected fields, from_value(s), to_value, and reason
4. Re-run repair and validation to verify

Example:
```python
'pattern9_new_pattern': {
    'fields': ['a99_new_field'],
    'from_value': 'WRONG_VALUE',
    'to_value': 'CORRECT_VALUE',
    'reason': 'Explanation of why this repair is needed'
}
```
