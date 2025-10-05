# DPA Enforcement Data Pipeline

This repository organizes raw and AI-annotated GDPR enforcement decisions, providing a reproducible pipeline for parsing, cleaning, exploratory diagnostics, and policy analysis. The goal of the project is to develop academic insights into decision-making patterns of highest academic quality.

## Data Processing Workflow

### Overview
The pipeline processes GDPR enforcement decisions through a four-phase system:
1. **Phase 1 (Extraction)**: Parse AI-annotated responses into structured CSV
2. **Phase 2 (Validation)**: Validate all 77 fields against schema rules
3. **Phase 3 (Repair)**: Auto-fix common validation errors using pattern-based rules
4. **Phase 4 (Enrichment)**: Generate analyst-ready features, long tables, and graph exports
5. **Phase 5 (Analysis)**: Build similarity cohorts, compare layered factors, and estimate mixed-effects models

**See `SCRIPTS-README.md` for process details and `data-sources-readme.md` for per-file data descriptions.**

### Primary Data Source
**`/raw_data/AI_analysis/AI-responses.txt`** - The authoritative dataset containing:
- **768 AI-annotated enforcement decisions**
- Format: Delimited text with "Answer N: VALUE" structure
- 757 complete responses (77 answers each)
- 11 incomplete responses (73 answers, missing Questions 74-77)

Response format:
```
----- RESPONSE DELIMITER -----
ID: <case_identifier>
----- RESPONSE DELIMITER -----
Answer 1: <value>
Answer 2: <value>
...
Answer 77: <value>
----- RESPONSE DELIMITER -----
```

### Phase 1: Extraction

**Script:** `scripts/1_parse_ai_responses.py`

**Input:** `/raw_data/AI_analysis/AI-responses.txt` (768 responses)

**Outputs:**
- `/outputs/phase1_extraction/main_dataset.csv` - 757 valid responses (77 fields each)
- `/outputs/phase1_extraction/data_with_errors.csv` - 11 incomplete responses (73 answers, missing Q74-77)
- `/outputs/phase1_extraction/extraction_log.txt` - Processing summary

**Field Mapping:** Answer N → aN_<field_name> (per schema)

**Command:**
```bash
python3 scripts/1_parse_ai_responses.py
```

### Phase 2: Validation

**Script:** `scripts/2_validate_dataset.py`

**Input:** `/outputs/phase1_extraction/main_dataset.csv` (757 rows)

**Outputs:**
- `/outputs/phase2_validation/validated_data.csv` - Clean rows (passed all validation)
- `/outputs/phase2_validation/validation_errors.csv` - Detailed error report
- `/outputs/phase2_validation/validation_report.txt` - Summary statistics

**Validation Coverage:**
- **All 77 fields** validated with 8 validation types:
  - ENUM (62 fields): Exact case-sensitive matching
  - INTEGER_RANGE (3 fields): Numeric bounds checking
  - INTEGER_OR_SENTINEL (1 field)
  - INTEGER_OR_SENTINELS (1 field)
  - FREE_TEXT (3 fields): Non-empty requirement
  - FREE_TEXT_OR_SENTINEL (4 fields)
  - SEMICOLON_LIST_OR_SENTINEL (2 fields)
  - SEMICOLON_INTEGERS_OR_SENTINELS (1 field)
- **8 conditional cross-field rules** enforced

**Commands:**
```bash
# Validate main dataset
python3 scripts/2_validate_dataset.py

# Validate custom dataset
python3 scripts/2_validate_dataset.py --input outputs/phase3_repair/repaired_dataset.csv
```

### Phase 2 Utility: Enum Analysis

**Script:** `scripts/2_analyze_enum_values.py`

**Purpose:** Analyze frequency of all enum field values to identify common mismappings

**Input:** `/outputs/phase1_extraction/main_dataset.csv` (or custom path)

**Outputs:**
- `/outputs/phase2_validation/enum_analysis.txt` - Human-readable report with ✓/✗ markers
- `/outputs/phase2_validation/enum_analysis.csv` - Machine-readable frequency data

**Command:**
```bash
python3 scripts/2_analyze_enum_values.py
```

### Phase 3: Repair

**Script:** `scripts/3_repair_data_errors.py`

**Purpose:** Automatically fix common validation errors using 5 targeted repair patterns and 1 manual-review flag

**Input:**
- `/outputs/phase1_extraction/main_dataset.csv` (data to repair)
- `/outputs/phase2_validation/validation_errors.csv` (errors to fix)

**Outputs:**
- `/outputs/phase3_repair/repaired_dataset.csv` - Repaired data
- `/outputs/phase3_repair/repair_log.txt` - Detailed repair log

**Repair Patterns & Flags:**
1. NOT_DISCUSSED → NO (two-option YES/NO fields)
2. NOT_APPLICABLE → NOT_DISCUSSED (discussion/violation fields that accept NOT_DISCUSSED)
3. BREACHED → YES (rights violation fields)
4. NOT_BREACHED → NO (rights violation fields)
5. Flag sector labels in `a8_defendant_class` for manual review (no automatic remap)
6. INTENTIONAL → AGGRAVATING (Art 83 factors)

> Binary fields (YES/NO only) are intentionally left for manual review if annotators supplied `NOT_APPLICABLE`.

**Financial currency support:** Both `a55_fine_currency` and `a58_turnover_currency` now accept `USD` and `CHF` alongside existing EEA codes. The repair script no longer rewrites non-EU currencies; review these rows manually if downstream analysis expects consolidation.

**Command:**
```bash
python3 scripts/3_repair_data_errors.py
```

### Phase 4: Enrichment & Delivery

**Script:** `scripts/4_enrich_prepare_outputs.py`

**Purpose:** Transform the repaired master dataset into a rich analytical bundle with ready-to-plot features, long tables, and graph exports.

**Input:**
- `/outputs/phase3_repair/repaired_dataset.csv` (default)
- FX reference: `/raw_data/reference/fx_rates.csv`
- Inflation reference: `/raw_data/reference/hicp_ea19.csv`

**Outputs (`/outputs/phase4_enrichment/`):**
- `0_fx_conversion_metadata.csv` – FX lookup diagnostics (method, source year/month, fallback flags) alongside nominal and converted amounts for fines and turnover.
- `0_fx_missing_review.csv` – Manual-review queue for cases where an FX conversion was requested but no rate was resolved.
- `1_enriched_master.csv` – 200+ engineered fields spanning temporal structure, monetary normalization (nominal & 2025 EUR), Art. 5 & 83 scoring, sanction profiles, OSS geography, QA flags, and text metadata.
- `2_processing_contexts.csv` – Long form processing contexts with ordering metadata.
- `3_vulnerable_groups.csv` – Exploded list of vulnerable data subjects per decision.
- `4_guidelines.csv` – Guidelines cited by each decision.
- `5_articles_breached.csv` – Parsed GDPR article references with numeric anchors.
- `graph/` – Neo4j-friendly node/edge CSVs linking decisions to authorities, defendants, articles, guidelines, and processing contexts.

**Command:**
```bash
python scripts/4_enrich_prepare_outputs.py
```

### Phase 5: Cohort Analysis & Modelling

**Script:** `scripts/5_analysis_similarity.py`

**Purpose:** Establish article-based similarity cohorts and evaluate how contexts, legal bases, defendant traits, and geography drive enforcement outcomes.

**Inputs:**
- `/outputs/phase4_enrichment/1_enriched_master.csv`

**Outputs (`/outputs/phase5_analysis/`):**
- `0_case_level_features.csv` – case-level dataset with parsed article sets, measure sets, context strata, and log-fine metrics.
- `1_baseline_article_cohorts.csv` – exact article-set cohorts with outcome dispersion and sanction profiles.
- `2_case_level_with_components.csv` / `2_relaxed_article_components.csv` – relaxed cohorts using Jaccard ≥ 0.8 unions.
- `3_context_effects.csv` – stratified comparisons of context flags (e.g., CCTV vs non-CCTV) within article cohorts.
- `3_legal_basis_effects.csv` – contrasts for Art. 6 statuses with `NOT_DISCUSSED` retained as its own bin.
- `3_defendant_type_effects.csv` – private vs public comparisons holding context combinations constant using flag-based context membership (no substring matching).
- `4_cross_country_pairs.csv` / `4_cross_country_summary.csv` – nearest-neighbour matches across countries with paired outcome stats.
- `5_mixed_effects_results.csv` / `5_mixed_effects_summary.txt` – mixed-effects regression with a random intercept for article set and variance component for country.
- `6_relaxed_cohort_contrasts.csv` – sensitivity of context effects under relaxed article components.
- `6_time_controls_summary.csv` – period splits (pre-2021 vs 2021+) for cohort-level averages.

**Command:**
```bash
python scripts/5_analysis_similarity.py
```

> ℹ️ The mixed-effects models emit warnings about singular covariance matrices; this reflects sparse article cohorts rather than runtime failure. Coefficients are retained for transparency.

### Running the Complete Pipeline

```bash
# Phase 1: Extract from AI responses (768 → 757 valid)
python3 scripts/1_parse_ai_responses.py

# Phase 2: Validate against schema (initial: 568/757 valid = 75.0%)
python3 scripts/2_validate_dataset.py

# Phase 2 Utility: Analyze enum patterns (identifies 1,421 invalid values)
python3 scripts/2_analyze_enum_values.py

# Phase 3: Repair common errors (see repair_log for current repair vs. flag breakdown)
python3 scripts/3_repair_data_errors.py

# Re-validate repaired data (final: 623/757 valid = 82.3%)
python3 scripts/2_validate_dataset.py --input outputs/phase3_repair/repaired_dataset.csv
# Produces: repaired_dataset_validated.csv, repaired_dataset_validation_errors.csv, repaired_dataset_validation_report.txt
```

## Data Quality Summary

**Extraction (Phase 1):**
- Total responses: 768
- Complete responses: 757/768 (98.6%)
- Incomplete responses: 11/768 (1.4%, missing Questions 74-77)

**Validation (Phase 2 - Initial):**
- Input rows: 757
- Clean rows: 568/757 (75.0%)
- Rows with errors: 189/757 (25.0%)
- Total validation errors: 1,511 (ERRORS: 1,144, WARNINGS: 367)

**Repair (Phase 3):**
- Rows repaired: 261/757 (34.5%)
- Total automatic repairs (latest run pre-flag update): 805 across 6 patterns; updated pipeline now logs manual-review flags for sector-class mismatches in addition to these repairs
- Most common fixes: NOT_DISCUSSED → NO, NOT_APPLICABLE → NOT_DISCUSSED, BREACHED → YES

**Validation (Phase 2 - Post-Repair):**
- Input rows: 757
- Clean rows: 623/757 (82.3%)
- Rows with errors: 134/757 (17.7%)
- Total validation errors: 1,288 (ERRORS: 921, WARNINGS: 367)
- **Improvement: +55 valid rows (+7.3%), -223 errors (-14.8%)**

## Schema Compliance

All data validated against:
- `/schema/main-schema-critically-important.md` - Field definitions and allowed values (77 fields: a1-a77)
- `/schema/AI-prompt-very-important.md` - Question-answer mapping (Questions 1-77)

**CRITICAL:** The schema is the authoritative source for all field validation rules.

## For Analysis

**Recommended Dataset:** `/outputs/phase3_repair/repaired_dataset_validated.csv`
- 623 rows that passed all 77 field validations (post-repair)
- Rows that satisfy all 8 conditional cross-field rules
- Ready for statistical analysis and policy research
- 82.3% of original dataset (623/757)

**Alternative (Pre-Repair):** `/outputs/phase2_validation/validated_data.csv`
- 568 rows (75.0% of original dataset)
- Use if you prefer unmodified data despite validation errors

## Research Tasks (Phase 5+ analytics)

Dedicated research scripts extend the pipeline using the enriched master dataset. Task 0 (`scripts/rt0_sanity_check.py`) performs type enforcement and data readiness checks, while Task 1 (`scripts/rt1_sanctions_architecture.py`) analyses sanction incidence, sanction mix indices, trigger/OSS deltas, and Art. 58 measure co-occurrence with bootstrap confidence intervals. Run `python run_research_tasks.py` to execute completed tasks sequentially; outputs are stored under `outputs/research_tasks/`.

## Directory Structure

```
/raw_data/
├── full-decisions/          # Raw enforcement decisions (.md files)
│   └── [authority]/[country]/[decision].md
└── AI_analysis/
    ├── AI-responses.txt     # PRIMARY DATA SOURCE (768 responses)
    ├── AI-prompt-very-important.md
    └── AI-full-responses.json

/outputs/
├── phase1_extraction/       # Parsed CSV data
│   ├── main_dataset.csv     # 757 valid responses (77 fields each)
│   └── extraction_log.txt
├── phase2_validation/       # Validation results
│   ├── validated_data.csv   # 568 clean rows (pre-repair)
│   ├── validation_errors.csv
│   ├── validation_report.txt
│   ├── enum_analysis.txt    # Enum frequency analysis
│   └── enum_analysis.csv
└── phase3_repair/           # Repaired and re-validated data
    ├── repaired_dataset.csv                 # 757 rows with 805 repairs applied (latest run)
    ├── repaired_dataset_validated.csv       # 623 clean rows → RECOMMENDED FOR ANALYSIS
    ├── repaired_dataset_validation_errors.csv
    ├── repaired_dataset_validation_report.txt
    └── repair_log.txt

/schema/
├── main-schema-critically-important.md  # Field definitions (a1-a77)
└── AI-prompt-very-important.md          # Question mapping (Q1-Q77)

/scripts/
├── 1_parse_ai_responses.py      # Phase 1: Extract to CSV
├── 2_validate_dataset.py        # Phase 2: Validate all fields
├── 2_analyze_enum_values.py    # Phase 2 Utility: Enum analysis
└── 3_repair_data_errors.py     # Phase 3: Auto-repair errors

/
├── readme.md                # This file - workflow overview
├── SCRIPTS-README.md        # Comprehensive script documentation
└── CLAUDE.md                # Guide for Claude Code instances
```

## Development Guidelines

1. **Consistency:** When designing new scripts, use existing scripts as reference. Avoid duplication.
2. **Phase Organization:** Scripts must follow phase numbering (1_*, 2_*, 3_*). Outputs go to `/outputs/phase[N]_[name]/`
3. **Schema First:** Always consult the schema before implementing features. Double-check conformance.
4. **Update Documentation:** Update `readme.md`, `SCRIPTS-README.md`, and `CLAUDE.md` when adding new phases.
5. **Data Transformations:** Document all data modifications in `SCRIPTS-README.md` with before/after examples.

## Additional Documentation

- **`SCRIPTS-README.md`** - Comprehensive documentation of all scripts with detailed data transformation examples
- **`CLAUDE.md`** - Guide for future Claude Code instances working in this repository
- **`schema/main-schema-critically-important.md`** - Authoritative field definitions (77 fields: a1-a77)
- **`schema/AI-prompt-very-important.md`** - AI annotation questionnaire (Questions 1-77)
- **`docs/phase4_enrichment_reference.md`** - Deep dive into Phase 4 helpers, derived fields, and output tables
- **`docs/phase4_improvement_plan.md`** - Ranked backlog of recommended enhancements for the enrichment stage
