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
   - `Answer 2` → `a2_case_name`
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
   - Example: `a2_case_name` can be any string
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
   - Example: `a77_articles_breached` format: "Art. 5(1)(a); Art. 6(1)"
   - Validates semicolon-separated article references

**Conditional Cross-Field Rules (8 Rules):**

1. `a19_art33_breached = NOT_APPLICABLE` when `a18_art33_discussed ≠ YES`
2. `a20_breach_notification_effect = NOT_DISCUSSED` when `a18_art33_discussed ≠ YES`
3. `a29_vulnerable_groups` should be empty when `a28_art9_discussed ≠ YES`
4. `a36_art6_legal_basis_other` should have content when any Art 6 basis = INVALID
5. `a13_sector_other` should have content when `a12_sector = OTHER`
6. `a54_fine_amount` should be > 0 when `a53_fine_imposed = YES`
7. `a55_fine_currency ≠ NOT_APPLICABLE` when `a53_fine_imposed = YES`
8. `a73_oss_role = NOT_APPLICABLE` when `a72_cross_border_oss ≠ YES`

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

**Repair Rules (6 Patterns):**

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

### Pattern 5: `[Sector Values] → OTHER`
**Field (1):** a8_defendant_class

**Values Changed:** MEDIA, EDUCATION, JUDICIAL, TELECOM → OTHER

**Reason:** Sector values incorrectly used in defendant classification field

**Example:**
- Before: `a8_defendant_class = MEDIA`
- After: `a8_defendant_class = OTHER`

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
4. Logs every repair with row ID, field, old value, new value

**Data Modifications:**
- Changes enum values to schema-compliant alternatives
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
**Result:** Applies 805 repairs to 261 rows (latest run; see repair_log for pattern mix)

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
4. **a55_fine_currency:** Added 'ISK' (Icelandic Króna) and 'USD' (US Dollar)
5. **a58_turnover_currency:** Added 'ISK' (Icelandic Króna) and 'USD' (US Dollar)

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
