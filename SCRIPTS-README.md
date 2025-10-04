# Data Pipeline Scripts Documentation

This document provides comprehensive documentation for all three phases of the GDPR enforcement data pipeline.

---

## Phase 1: Data Extraction

### Script: `scripts/1_parse_ai_responses.py`

**Purpose:** Parse AI-generated responses from delimited text format into structured CSV.

**Input:**
- `/raw_data/AI_analysis/AI-responses.txt` (768 responses in delimited text format)

**Output:**
- `/outputs/phase1_extraction/main_dataset.csv` (757 rows × 77 fields)
- `/outputs/phase1_extraction/extraction_log.txt` (parsing statistics)

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
   - Successfully parsed: 757/768 responses (98.6%)
   - Incomplete/skipped: 11 responses (1.4%)

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

**Validation Rules (8 Types):**

1. **`enum`** - Must match allowed values exactly
   - Example: `a1_country_code` must be in [AT, BE, BG, ..., EL, IS, ...]
   - Generates ERROR if value not in allowed list

2. **`integer_range`** - Must be valid integer within range
   - Example: `a4_decision_year` must be 2018-2025
   - Generates ERROR if non-integer or outside range
   - Generates WARNING if `warn_outside` flag set

3. **`float_positive`** - Must be positive number
   - Example: `a54_fine_amount` must be ≥ 0
   - Generates ERROR if negative or non-numeric

4. **`free_text`** - Allows any text (length validation only)
   - Example: `a2_authority_name` can be any string
   - Generates WARNING if exceeds 500 characters

5. **`delimited_list`** - Semicolon-separated enum values
   - Example: `a14_processing_contexts` must be list of valid contexts
   - Validates each item against allowed values
   - Generates ERROR if any item invalid

6. **`conditional`** - Value depends on other field
   - Example: `a73_oss_role` must be NOT_APPLICABLE when `a72_cross_border_oss ≠ YES`
   - Generates ERROR if condition violated

7. **`optional_free_text`** - Text field that's conditional
   - Example: `a13_sector_other` should have content when `a12_sector = OTHER`
   - Generates WARNING if missing when expected

8. **`articles_list`** - Special format for GDPR article citations
   - Example: `a77_articles_breached` canonical format `5;6;15;17`
   - Validator auto-strips prefixes like "Art."/"Article" and `(1)(a)` suffixes before checking numeric anchors

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
     ✓ NOT_DISCUSSED                    564
     ✓ YES                              136
     ✓ NO                                 3

   Invalid values (NOT IN SCHEMA):
     ✗ BREACHED                          47  ← FIX NEEDED
     ✗ NOT_APPLICABLE                     6  ← FIX NEEDED

   Summary: 53/757 (7.0%) values are invalid
   ```

3. **CSV Output:**
   ```
   field_name,value,count,valid,status
   a43_transparency_violated,BREACHED,47,NO,✗ FIX
   ```

**No Data Modification:** This is a read-only analysis utility.

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
- `/outputs/phase3_repair/repaired_dataset.csv` (repaired data)
- `/outputs/phase3_repair/repaired_dataset_validated.csv` (rows that pass all checks post-repair)
- `/outputs/phase3_repair/repaired_dataset_validation_errors.csv`
- `/outputs/phase3_repair/repaired_dataset_validation_report.txt`
- `/outputs/phase3_repair/repair_log.txt` (detailed repair log)

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

**Usage:**
```bash
python3 scripts/3_repair_data_errors.py
```

---

## Complete Pipeline Workflow

### Step 1: Extract
```bash
python3 scripts/1_parse_ai_responses.py
```
**Result:** 768 → 757 rows (98.6% extraction rate)

### Step 2: Validate (Initial)
```bash
python3 scripts/2_validate_dataset.py
```
**Result:** 568/757 valid rows (75.0%)

### Step 3: Analyze Patterns
```bash
python3 scripts/2_analyze_enum_values.py
```
**Result:** Identifies 1,421 invalid enum values across 62 fields

### Step 4: Repair
```bash
python3 scripts/3_repair_data_errors.py
```
**Result:** Applies hundreds of targeted repairs per run and logs any outstanding flags (see `repair_log.txt` for the precise mix).

### Step 5: Validate (Post-Repair)
```bash
python3 scripts/2_validate_dataset.py --input outputs/phase3_repair/repaired_dataset.csv
```
**Result:** 623/757 valid rows (82.3%) - **+55 rows improved, +7.3%**

---

## Data Quality Metrics

| Metric | Initial | After Phase 3 | Change |
|--------|---------|---------------|--------|
| **Valid Rows** | 568/757 (75.0%) | 623/757 (82.3%) | +55 rows (+7.3%) |
| **Total Errors** | 1,511 | 1,288 | -223 errors (-14.8%) |
| **Schema Violations (ERROR)** | 1,144 | 921 | -223 errors (-19.5%) |
| **Suspicious Values (WARNING)** | 367 | 367 | 0 (±0%) |

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
- Re-validation errors: `/outputs/phase3_repair/validation_errors.csv`
- Re-validation report: `/outputs/phase3_repair/validation_report.txt`

---

## Phase 4: Enrichment & Multi-format Delivery

### Script: `scripts/4_enrich_prepare_outputs.py`

**Purpose:** Convert the repaired Phase 3 dataset into a feature-complete analytical package and graph-friendly extracts.

**Inputs:**
- `/outputs/phase3_repair/repaired_dataset.csv`
- FX lookups: `/raw_data/reference/fx_rates.csv`
- HICP deflators: `/raw_data/reference/hicp_ea19.csv`

**Outputs:**
- `/outputs/phase4_enrichment/1_enriched_master.csv` – Adds temporal fields, FX/EUR normalization (nominal & real 2025 euros), sanction and Art. 83 scoring, contextual flags, OSS geography, QA signals, and keyword metadata.
- `/outputs/phase4_enrichment/2_processing_contexts.csv` – Long table of processing contexts with positional order.
- `/outputs/phase4_enrichment/3_vulnerable_groups.csv` – Exploded vulnerable group annotations.
- `/outputs/phase4_enrichment/4_guidelines.csv` – Guidelines referenced per decision.
- `/outputs/phase4_enrichment/5_articles_breached.csv` – Parsed GDPR articles with numeric anchors and positions.
- `/outputs/phase4_enrichment/graph/` – Neo4j bulk-import node and edge CSVs for Decisions, Authorities, Defendants, Articles, Guidelines, and Contexts.

For a step-by-step explanation of each enrichment helper and derived column family, consult
`docs/phase4_enrichment_reference.md`.

**Key Transformations:**

1. **Temporal normalization** – Derives `decision_year`, `decision_month`, inferred dates, quarter buckets, and granularity flags even when only the year is known.
2. **Monetary harmonization** – Maps all fines/turnover to EUR using ECB-informed FX tables, adds deflated 2025 EUR values, log scaling, ratio metrics, and categorical buckets.
3. **Rights & breaches** – Computes Art. 5 boolean flags, rights violation profiles, per-article presence flags, and priority breach families.
4. **Sanctions & Article 83 scoring** – Produces sanction profiles, measure counts, warning/fine convenience booleans, and numerical Art. 83 factor scores with coverage counts.
5. **Contextual cross-features** – Expands processing contexts into binary indicators, builds `sector_x_context_key`, and surfaces `context_profile` strings.
6. **Quality and QA flags** – Highlights sector detail gaps, currency omissions, Article 33 inconsistencies, and systematic-factor mismatches.
7. **Graph exports** – Emits node/edge CSVs aligning with the schema described in the analyst brief for direct Neo4j bulk import or further network analysis.

**Usage:**
```bash
python scripts/4_enrich_prepare_outputs.py
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
