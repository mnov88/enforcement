# Repository Guidelines

## Project Structure & Module Organization
- `scripts/AIscript.py` holds the Streamlit batch processor; reuse its helper functions when adding more pipeline steps.
- `raw_data/` contains authoritative datasets (`main_dataset.csv`, `data_with_errors.csv`, AI prompt/responses) and must remain read-only during processing.
- `schema/` stores the canonical field definitions; reconcile every new column or enum against these docs before writing code.
- Create phase-specific result folders under `outputs/<phase_name>/` when persisting derived data; keep filenames prefixed with execution order (e.g., `1_clean_...`).

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` – isolate dependencies.
- `pip install streamlit pandas aiohttp requests` – minimal runtime stack for `AIscript.py`.
- `streamlit run scripts/AIscript.py` – launch the UI for batch annotation.
- `pytest` (once tests exist in `tests/`) – run unit checks on parsing utilities.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and descriptive snake_case for functions, columns, and output filenames.
- Keep async helpers pure and stateless; prefer type hints and docstrings mirroring `OpenRouterAPI`.
- Centralize configuration constants near the top of a module; avoid hard-coding API details deeper in the call stack.

## Testing Guidelines
- Validate new helpers with lightweight `pytest` cases under `tests/`, naming files `test_<feature>.py`.
- Use small CSV fixtures that mimic `schema/main-schema-critically-important.md` to assert column presence and enum validation.
- Before large runs, exercise the UI with the built-in sample dataset to confirm rate-limit messaging and output paths.

## Commit & Pull Request Guidelines
- Write imperative, scoped commit messages (`Add preset override validation`) and keep data updates isolated from code changes.
- PRs should link any related research tickets, describe input/output files touched, and attach UI screenshots when Streamlit components change.
- Document phase additions in `readme.md` while the PR is open to preserve pipeline traceability.

## Security & Configuration Tips
- Never commit OpenRouter API keys; store them in environment variables or Streamlit secrets and use the sidebar validator.
- Scrub outputs before publishing to ensure no personal data from `raw_data/` escapes the workspace.
