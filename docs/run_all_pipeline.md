# Run-All Pipeline Script

This guide explains how to use `scripts/run_all_pipeline.py` to execute the
entire GDPR enforcement data pipeline (Phases 1–4) in a single command. The
script orchestrates the existing phase helpers without duplicating their
logic, ensuring that each stage reads the exact output from the previous one.

## Key Features

- Accepts either a path to an `AI-responses.txt` style file or raw response
  text supplied via the `--input-text` flag.
- Reuses the phase modules directly (`run_phase1`, `run_phase2`,
  `run_phase3`, `enrich_dataset`) so the data flow mirrors the documented
  pipeline.
- Wraps each phase call in guarded error handling so missing inputs or
  unexpected IO problems surface with actionable messages.
- Writes all artefacts to the standard `/outputs/phase*_*/` directories by
  default, while allowing overrides through `--output-root`.
- Automatically re-validates the repaired dataset before triggering Phase 4
  enrichment.

## Usage

```bash
python scripts/run_all_pipeline.py \
  --input-file raw_data/AI_analysis/AI-responses.txt
```

### Optional Flags

| Flag | Purpose |
| --- | --- |
| `--input-text "…"` | Provide raw AI response text directly. The script creates a temporary file and removes it after the run (unless `--keep-temp-input` is set). |
| `--output-root <path>` | Store all phase outputs under a different root directory while retaining per-phase subfolders. |
| `--skip-enrichment` | Execute only Phases 1–3 and skip the enrichment helpers. |
| `--keep-temp-input` | Preserve the temporary file generated from `--input-text`. |

## Phase-by-Phase Workflow

1. **Phase 1 – Extraction**
   - Calls `run_phase1` from `scripts/1_parse_ai_responses.py`.
   - Input: AI response text (from `--input-file` or the temporary file created for `--input-text`).
   - Output: `/outputs/phase1_extraction/main_dataset.csv` and supporting logs/errors inside the same folder.

2. **Phase 2 – Validation**
   - Calls `run_phase2` from `scripts/2_validate_dataset.py` on the Phase 1 CSV.
   - Output: `/outputs/phase2_validation/validated_data.csv`, `validation_errors.csv`, and `validation_report.txt`.
   - The generated `validation_errors.csv` is passed directly into Phase 3 when present.

3. **Phase 3 – Repair**
   - Invokes `run_phase3` from `scripts/3_repair_data_errors.py` using the Phase 1 dataset plus the Phase 2 error ledger.
   - Output: `/outputs/phase3_repair/repaired_dataset.csv` and `repair_log.txt`.
   - Immediately re-validates the repaired dataset, producing `repaired_dataset_validated.csv` (clean subset) and updated validation artefacts in the same folder.
   - The run halts if the validation ledger predates the dataset or references unknown IDs, preventing stale repairs.

4. **Phase 4 – Enrichment (optional)**
   - Unless `--skip-enrichment` is supplied, calls `enrich_dataset` from `scripts/4_enrich_prepare_outputs.py`.
   - Input: `/outputs/phase3_repair/repaired_dataset.csv` plus reference tables in `raw_data/reference/` (FX, HICP, context taxonomy, region map).
   - Output: `/outputs/phase4_enrichment/` bundle (master dataset, long tables, and graph exports).

The script prints a concise summary of the generated files at the end of the run so you can quickly locate the artefacts for downstream analysis.

## Tips

- The underlying modules enforce the schema documented in
  `schema/main-schema-critically-important.md`, so schema updates will flow
  through the pipeline automatically.
- When experimenting with alternate output locations (via `--output-root`),
  remember that Phase 4 still expects the FX and HICP references from
  `raw_data/reference/`.
- For iterative debugging, use `--skip-enrichment` to accelerate runs and
  inspect the repair + revalidation outputs before generating the enriched
  bundle.
