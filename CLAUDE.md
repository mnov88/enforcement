# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a data pipeline for analyzing GDPR enforcement decisions. The project extracts structured data from legal decisions using AI annotation, validates responses against a strict schema, and prepares datasets for academic policy analysis.

**Core objective:** Develop academic insights into GDPR enforcement patterns with highest academic quality.

## Key Architecture

### Data Flow (3-Phase Pipeline)

**Phase 1: Extraction**
1. **Input:** `/raw_data/AI_analysis/AI-responses.txt` (768 AI-annotated responses)
2. **Script:** `scripts/1_parse_ai_responses.py`
3. **Output:** `/outputs/phase1_extraction/main_dataset.csv` (757 rows × 77 fields)

**Phase 2: Validation**
1. **Input:** `/outputs/phase1_extraction/main_dataset.csv`
2. **Script:** `scripts/2_validate_dataset.py`
3. **Output:** `/outputs/phase2_validation/validated_data.csv` (623 clean rows after repair)

**Phase 3: Repair**
1. **Input:** `/outputs/phase1_extraction/main_dataset.csv` + validation errors
2. **Script:** `scripts/3_repair_data_errors.py` (6 pattern-based repair rules)
3. **Output:** `/outputs/phase3_repair/repaired_dataset.csv` (repaired rows) + `/outputs/phase3_repair/repaired_dataset_validated.csv` (623-row clean subset)

**Phase 2 Utility:** `scripts/2_analyze_enum_values.py` - Enum frequency analysis for identifying repair patterns

### Schema-Driven Development

**THE SCHEMA IS LAW.** Before implementing any feature:

1. Read `/schema/main-schema-critically-important.md` - defines 77 fields (a1-a77) with exact enum values
2. Cross-reference `/schema/AI-prompt-very-important.md` - contains the questionnaire that generates responses
3. Validate all field names, allowed values, and format rules (e.g., semicolon-separated lists, quote-escaping for commas)

The schema defines:
- Basic metadata (country codes, dates, authority names)
- Defendant information (classification, size, sector)
- GDPR violations (Articles 5, 6, 15-22, 33)
- Article 58(2) corrective measures
- Article 83(2) penalty factors (AGGRAVATING/MITIGATING/NEUTRAL/NOT_DISCUSSED)
- Financial penalties and turnover data
- Financial currency enums accept EUR/GBP/… plus USD; do not auto-convert values in scripts.
- Cross-border processing (One-Stop-Shop)

### Phase-Based Organization

Scripts and outputs follow a phase structure:

- Script filenames: `<phase_number>_<description>.py` (e.g., `1_parse_ai_responses.py`, `2_validate_dataset.py`)
- Outputs: `/outputs/<phase_name>/` folders (e.g., `/outputs/phase1_extraction/`, `/outputs/phase2_validation/`)
- Update `readme.md` and `SCRIPTS-README.md` when adding new phases or modifying behavior

**Current Pipeline:**
1. **Phase 1 (Extraction):** Parse delimited text → CSV with 77 fields
2. **Phase 2 (Validation):** Validate all fields against schema (8 validation types, 8 cross-field rules)
3. **Phase 3 (Repair):** Auto-fix common errors using 6 pattern-based rules

See `SCRIPTS-README.md` for comprehensive documentation of all data transformations.

## Development Commands

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install streamlit pandas aiohttp requests
```

### Running the AI Batch Processor
```bash
streamlit run scripts/AIscript.py
```

**Key features:**
- Upload CSV with ID + content columns
- Configure OpenRouter presets or direct model selection
- Row selection (1-based): `2-999,5,10-12` for targeted processing
- Rate limiting: 20 RPM for free models (configurable)
- Chunk-based processing with auto-resume
- Outputs: CSV, JSON, ZIP of markdown files, plain-text responses

### Testing
```bash
pytest  # Once tests exist in tests/
```

## Important Constraints

### Data Integrity

- **NEVER modify files in `/raw_data/` during processing** - treat as read-only
- All derived data goes to `/outputs/<phase_name>/`
- Use the schema for all column validation before writing code

### Script Reusability

- Check existing scripts before creating new ones - avoid duplication
- Reuse helper functions from `scripts/AIscript.py`:
  - `OpenRouterAPI` class for API interactions
  - `parse_row_selection()` for row parsing
  - `save_response_as_markdown()` for file persistence
  - `build_responses_text()` for plain-text exports

### Data Pipeline Context

**Critical Fix Applied:** Original parser had regex bug requiring double delimiter before ID line, causing 50% data loss. Fixed to single delimiter pattern, achieving 98.6% extraction rate (757/768).

**AI Response Format:**
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

**Field Mapping:** Answer N → aN_field_name (e.g., Answer 1 → a1_country_code)

**Data Quality:** 82.3% of extracted rows pass all validation checks after Phase 3 repair (623/757 valid rows)

## API Configuration

### OpenRouter Integration

- **Free tier limits:** 20 RPM, 50 requests/day (<$10 credits) or 1000/day ($10+ credits)
- **Preset support:** Use `@preset/<slug>` for consistent model/temperature/system prompts
- **Preset methods:**
  - `field`: `{"preset": "preset-name"}` (recommended)
  - `direct_reference`: `"@preset/preset-name"` as model parameter
- **Recommended models for legal extraction:**
  - DeepSeek R1 Distill 70B (reasoning, temp 0.1)
  - Qwen 2.5 Coder 32B (structured extraction, temp 0.2)
  - DeepSeek V3.1 (hybrid reasoning, temp 0.1)
  - Grok 4 Fast (2M context for large documents, temp 0.2)

### Rate Limit Management

The `OpenRouterAPI` class implements token bucket pacing:
- Pre-fills bucket with RPM limit tokens
- Refills at `60/RPM` second intervals
- Concurrent request limiter (`asyncio.Semaphore`) respects both RPM and max concurrency
- Set `use_internal_rate_limit=False` when using external rate limiters

## Coding Standards

- Follow PEP 8 with 4-space indentation
- Use snake_case for functions, variables, column names
- Type hints and docstrings required for all functions
- Keep async helpers pure and stateless
- Centralize configuration constants at module top
- Field names must match schema exactly (e.g., `a36_legal_basis_summary`, not `legal_basis_summary`)

## Git Workflow

- Commit messages: Imperative mood, scoped (`Add preset override validation`)
- Keep data updates separate from code changes
- Never commit API keys - use environment variables or Streamlit secrets
- Document phase additions in `readme.md` before merging PRs
