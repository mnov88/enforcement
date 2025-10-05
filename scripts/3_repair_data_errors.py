#!/usr/bin/env python3
"""
Phase 3: Auto-Repair Common Data Errors

Automatically fixes targeted validation errors identified in Phase 2 while
preserving auditability:
1. Sentinel coercions (e.g., NOT_DISCUSSED → NO) are tracked in
   `phase3_coercion_flags` rather than silently overwriting uncertainty.
2. Schema hashes from Phase 2 are verified before repairs run and are
   embedded into Phase 3 artefacts for provenance.

Input:  /outputs/phase1_extraction/main_dataset.csv
        /outputs/phase2_validation/validation_errors.csv
Output: /outputs/phase3_repair/repaired_dataset.csv
        /outputs/phase3_repair/repair_log.txt
        /outputs/phase3_repair/schema_snapshot.json
"""

import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, Optional

from schema_utils import (
    compute_schema_hash,
    read_schema_snapshot,
    resolve_schema_files,
    write_schema_snapshot,
)

# Define repair rules
REPAIR_RULES = {
    # Pattern 1: NOT_DISCUSSED → NO (two-option YES/NO fields)
    'pattern1_not_discussed_to_no': {
        'fields': [
            'a18_art33_discussed',
            'a28_art9_discussed',
            'a53_fine_imposed',
            'a56_turnover_discussed',
            'a70_systematic_art83_discussion'
        ],
        'from_value': 'NOT_DISCUSSED',
        'to_value': 'NO',
        'reason': 'Two-option field expects YES or NO, NOT_DISCUSSED invalid',
        'track_coercion': True,
    },

    # Pattern 2: NOT_APPLICABLE → NOT_DISCUSSED (fields that allow NOT_DISCUSSED)
    'pattern2_not_applicable_to_not_discussed': {
        'fields': [
            'a3_appellate_decision', 'a15_data_subject_complaint', 'a16_media_attention',
            'a17_official_audit', 'a20_breach_notification_effect',
            'a21_art5_lawfulness_fairness', 'a22_art5_purpose_limitation', 'a23_art5_data_minimization',
            'a24_art5_accuracy', 'a25_art5_storage_limitation', 'a26_art5_integrity_confidentiality',
            'a27_art5_accountability',
            'a30_art6_consent', 'a31_art6_contract', 'a32_art6_legal_obligation',
            'a33_art6_vital_interests', 'a34_art6_public_task', 'a35_art6_legitimate_interest',
            'a37_right_access_violated', 'a38_right_rectification_violated', 'a39_right_erasure_violated',
            'a40_right_restriction_violated', 'a41_right_portability_violated', 'a42_right_object_violated',
            'a43_transparency_violated', 'a44_automated_decisions_violated',
            'a45_warning_issued', 'a46_reprimand_issued', 'a47_comply_data_subject_order',
            'a48_compliance_order', 'a49_breach_communication_order', 'a50_erasure_restriction_order',
            'a51_certification_withdrawal', 'a52_data_flow_suspension',
            'a59_nature_gravity_duration', 'a60_intentional_negligent', 'a61_mitigate_damage_actions',
            'a62_technical_org_measures', 'a63_previous_infringements', 'a64_cooperation_authority',
            'a65_data_categories_affected', 'a66_infringement_became_known', 'a67_prior_orders_compliance',
            'a68_codes_certification', 'a69_other_factors',
            'a71_first_violation', 'a72_cross_border_oss'
        ],
        'from_value': 'NOT_APPLICABLE',
        'to_value': 'NOT_DISCUSSED',
        'reason': 'NOT_APPLICABLE used incorrectly; defer to NOT_DISCUSSED sentinel when value missing'
    },

    # Pattern 3: BREACHED → YES (rights violation fields)
    'pattern3_breached_to_yes': {
        'fields': [
            'a37_right_access_violated',
            'a38_right_rectification_violated',
            'a39_right_erasure_violated',
            'a40_right_restriction_violated',
            'a41_right_portability_violated',
            'a42_right_object_violated',
            'a43_transparency_violated',
            'a44_automated_decisions_violated'
        ],
        'from_value': 'BREACHED',
        'to_value': 'YES',
        'reason': 'Rights violation fields expect YES/NO/NOT_DISCUSSED, not BREACHED'
    },

    # Pattern 4: NOT_BREACHED → NO (rights violation fields)
    'pattern4_not_breached_to_no': {
        'fields': [
            'a37_right_access_violated',
            'a38_right_rectification_violated',
            'a39_right_erasure_violated',
            'a40_right_restriction_violated',
            'a41_right_portability_violated',
            'a42_right_object_violated',
            'a43_transparency_violated',
            'a44_automated_decisions_violated',
            'a19_art33_breached'
        ],
        'from_value': 'NOT_BREACHED',
        'to_value': 'NO',
        'reason': 'Field expects YES/NO format, not BREACHED/NOT_BREACHED'
    },

    # Pattern 6: INTENTIONAL → AGGRAVATING (Art 83 factors)
    'pattern6_intentional_to_aggravating': {
        'fields': ['a60_intentional_negligent'],
        'from_value': 'INTENTIONAL',
        'to_value': 'AGGRAVATING',
        'reason': 'Field expects AGGRAVATING/MITIGATING/NEUTRAL/NOT_DISCUSSED, not INTENTIONAL'
    }
}


FLAG_PATTERNS = {
    'pattern5_sector_in_defendant_class': {
        'field': 'a8_defendant_class',
        'flag_values': ['MEDIA', 'EDUCATION', 'JUDICIAL', 'TELECOM'],
        'reason': 'Sector-specific label captured in defendant_class; requires manual review'
    }
}


COERCION_NOTES_COLUMN = 'phase3_coercion_flags'


def verify_schema_alignment(error_file: str, project_root: Path) -> tuple[str, Path]:
    """Ensure the validation artefacts align with the current schema snapshot."""
    error_path = Path(error_file)
    snapshot_path = error_path.parent / 'schema_snapshot.json'

    snapshot = read_schema_snapshot(snapshot_path)
    if not snapshot or 'schema_hash' not in snapshot:
        raise RuntimeError(
            "Schema snapshot metadata missing alongside validation errors; rerun Phase 2."
        )

    expected_hash = snapshot['schema_hash']
    current_hash = compute_schema_hash(project_root)
    if expected_hash != current_hash:
        raise RuntimeError(
            "Schema hash mismatch between Phase 2 and current repository. "
            "Rerun validation before applying repairs."
        )

    return expected_hash, snapshot_path


def load_validation_errors(error_file: str) -> tuple[dict, Optional[str]]:
    """Load validation errors and group by row_id and field."""
    errors_by_row = defaultdict(lambda: defaultdict(list))
    observed_schema_hash: Optional[str] = None

    with open(error_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for error in reader:
            if error['severity'] == 'ERROR':  # Only fix ERRORS, not WARNINGS
                row_id = error['row_id']
                field = error['field_name']
                errors_by_row[row_id][field].append(error)

            schema_hash = error.get('schema_hash')
            if schema_hash:
                if observed_schema_hash is None:
                    observed_schema_hash = schema_hash
                elif observed_schema_hash != schema_hash:
                    raise RuntimeError(
                        "Validation errors reference multiple schema hashes; rerun Phase 2."
                    )

    return errors_by_row, observed_schema_hash


def apply_repairs(row: dict, row_id: str, errors_by_row: dict) -> tuple[dict, list[dict], list[str]]:
    """
    Apply automatic repairs to a row.
    Returns (repaired_row, actions_taken, coerced_fields)
    """
    repaired = row.copy()
    actions = []
    coerced_fields: list[str] = []

    # Get errors for this row
    row_errors = errors_by_row.get(row_id, {})

    # Apply each repair rule
    for pattern_name, rule in REPAIR_RULES.items():
        for field in rule['fields']:
            if field not in repaired:
                continue

            # Handle rules with single from_value
            if 'from_value' in rule:
                if repaired[field] == rule['from_value'] and field in row_errors:
                    repaired[field] = rule['to_value']
                    action = {
                        'field': field,
                        'from': rule['from_value'],
                        'to': rule['to_value'],
                        'pattern': pattern_name,
                        'reason': rule['reason'],
                        'action': 'REPAIR'
                    }
                    if rule.get('track_coercion'):
                        action['coercion'] = True
                        coerced_fields.append(field)
                    actions.append(action)

            # Handle rules with multiple from_values (e.g., pattern5)
            elif 'from_values' in rule:
                if repaired[field] in rule['from_values'] and field in row_errors:
                    old_value = repaired[field]
                    repaired[field] = rule['to_value']
                    actions.append({
                        'field': field,
                        'from': old_value,
                        'to': rule['to_value'],
                        'pattern': pattern_name,
                        'reason': rule['reason'],
                        'action': 'REPAIR'
                    })

    for pattern_name, rule in FLAG_PATTERNS.items():
        field = rule['field']
        if field not in repaired or field not in row_errors:
            continue
        if repaired[field] in rule['flag_values']:
            actions.append({
                'field': field,
                'from': repaired[field],
                'to': repaired[field],
                'pattern': pattern_name,
                'reason': rule['reason'],
                'action': 'FLAG'
            })

    return repaired, actions, coerced_fields


def repair_dataset(input_file: str, error_file: str, output_file: str, log_file: str):
    """Repair dataset and generate log."""

    project_root = Path(__file__).parent.parent

    print("Verifying schema snapshot...")
    schema_hash, snapshot_path = verify_schema_alignment(error_file, project_root)
    print(f"✓ Schema snapshot verified ({schema_hash})")

    # Load validation errors
    print("Loading validation errors...")
    errors_by_row, observed_schema_hash = load_validation_errors(error_file)
    print(f"✓ Loaded errors for {len(errors_by_row)} rows")
    if observed_schema_hash and observed_schema_hash != schema_hash:
        raise RuntimeError(
            "Validation errors were generated against a different schema hash; rerun Phase 2."
        )

    # Load and repair dataset
    print(f"\nRepairing dataset...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if COERCION_NOTES_COLUMN not in fieldnames:
        fieldnames.append(COERCION_NOTES_COLUMN)

    repaired_rows = []
    repairs_by_pattern = defaultdict(int)
    flags_by_pattern = defaultdict(int)
    coercions_by_field = defaultdict(int)
    rows_repaired = 0
    rows_flagged = 0
    coercion_rows = 0
    repair_details = []

    for row in rows:
        row.setdefault(COERCION_NOTES_COLUMN, '')
        row_id = row.get('id', 'unknown')
        repaired_row, actions, coerced_fields = apply_repairs(row, row_id, errors_by_row)

        if coerced_fields:
            coercion_rows += 1
            existing = [note.strip() for note in repaired_row.get(COERCION_NOTES_COLUMN, '').split(';') if note.strip()]
            combined = sorted(set(existing + coerced_fields))
            repaired_row[COERCION_NOTES_COLUMN] = ';'.join(combined)
            for field in coerced_fields:
                coercions_by_field[field] += 1
        else:
            repaired_row.setdefault(COERCION_NOTES_COLUMN, '')

        if actions:
            has_repair = any(action['action'] == 'REPAIR' for action in actions)
            has_flag = any(action['action'] == 'FLAG' for action in actions)

            if has_repair:
                rows_repaired += 1
            if has_flag:
                rows_flagged += 1

            for action in actions:
                if action['action'] == 'REPAIR':
                    repairs_by_pattern[action['pattern']] += 1
                elif action['action'] == 'FLAG':
                    flags_by_pattern[action['pattern']] += 1

                repair_details.append({
                    'row_id': row_id,
                    'field': action['field'],
                    'from_value': action['from'],
                    'to_value': action['to'],
                    'pattern': action['pattern'],
                    'action': action['action'],
                    'coercion': 'YES' if action.get('coercion') else '',
                })

        repaired_rows.append(repaired_row)

    # Write repaired dataset
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(repaired_rows)

    print(f"✓ Repaired {rows_repaired} rows")
    if rows_flagged:
        print(f"⚑ Flagged {rows_flagged} rows for manual review")
    if coercion_rows:
        print(f"✓ Recorded {coercion_rows} rows with coercion flags")
    total_actions = sum(repairs_by_pattern.values()) + sum(flags_by_pattern.values())
    print(f"✓ Total actions logged: {total_actions}")

    output_dir = Path(output_file).parent
    schema_files = resolve_schema_files(project_root)
    snapshot_out = write_schema_snapshot(output_dir, schema_hash, schema_files)
    print(f"✓ Recorded schema snapshot to {snapshot_out.name}")

    # Generate repair log
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Phase 3: Data Repair Log\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Input: {input_file}\n")
        f.write(f"Output: {output_file}\n")
        f.write(f"Schema hash: {schema_hash}\n")
        f.write(f"Validation snapshot: {snapshot_path}\n")
        f.write(f"Phase 3 snapshot: {snapshot_out}\n\n")

        f.write("REPAIR SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total rows processed: {len(rows)}\n")
        f.write(f"Rows repaired: {rows_repaired}\n")
        f.write(f"Rows flagged for manual review: {rows_flagged}\n")
        f.write(f"Rows with coercion flags: {coercion_rows}\n")
        total_actions = sum(repairs_by_pattern.values()) + sum(flags_by_pattern.values())
        f.write(f"Total actions logged: {total_actions}\n\n")

        if coercions_by_field:
            f.write("COERCION FLAGS BY FIELD\n")
            f.write("-" * 70 + "\n")
            for field, count in sorted(coercions_by_field.items()):
                f.write(f"{field}: {count}\n")
            f.write("\n")

        f.write("REPAIRS BY PATTERN\n")
        f.write("-" * 70 + "\n")
        if repairs_by_pattern:
            for pattern, count in sorted(repairs_by_pattern.items()):
                f.write(f"{pattern}: {count} repairs\n")
                rule = REPAIR_RULES[pattern]
                if 'from_value' in rule:
                    f.write(f"  Rule: {rule['from_value']} → {rule['to_value']}\n")
                elif 'from_values' in rule:
                    from_vals = ', '.join(rule['from_values'])
                    f.write(f"  Rule: [{from_vals}] → {rule['to_value']}\n")
                f.write(f"  Reason: {rule['reason']}\n\n")
        else:
            f.write("(no automatic repairs applied)\n\n")

        f.write("FLAGS BY PATTERN\n")
        f.write("-" * 70 + "\n")
        if flags_by_pattern:
            for pattern, count in sorted(flags_by_pattern.items()):
                f.write(f"{pattern}: {count} flags\n")
                rule = FLAG_PATTERNS[pattern]
                values = ', '.join(rule['flag_values'])
                f.write(f"  Values flagged: {values}\n")
                f.write(f"  Reason: {rule['reason']}\n\n")
        else:
            f.write("(no rows flagged for manual review)\n\n")

        f.write("DETAILED ACTIONS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Row ID, Field, From, To, Pattern, Action, Coercion\n")
        for detail in repair_details[:100]:  # First 100 for brevity
            f.write(
                f"{detail['row_id']}, {detail['field']}, {detail['from_value']}, {detail['to_value']}, "
                f"{detail['pattern']}, {detail['action']}, {detail['coercion']}\n"
            )

        if len(repair_details) > 100:
            f.write(f"... and {len(repair_details) - 100} more actions\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Repair complete. Run Phase 2 validation again to verify.\n")
        f.write("=" * 70 + "\n")

    total_actions = sum(repairs_by_pattern.values()) + sum(flags_by_pattern.values())
    return rows_repaired, total_actions, schema_hash


def run_phase3(
    input_file: Path,
    error_file: Path,
    output_file: Optional[Path] = None,
    log_file: Optional[Path] = None,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute phase 3 repairs with configurable paths."""

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if output_file is None:
        output_file = project_root / 'outputs' / 'phase3_repair' / 'repaired_dataset.csv'
    if log_file is None:
        log_file = output_file.parent / 'repair_log.txt'

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("Phase 3: Auto-Repair Data Errors")
        print("=" * 70)
        print(f"Input:  {input_file}")
        print(f"Errors: {error_file}")
        print(f"Output: {output_file}")
        print(f"Log:    {log_file}")
        print()

    rows_repaired, total_actions, schema_hash = repair_dataset(
        str(input_file), str(error_file), str(output_file), str(log_file)
    )

    if verbose:
        print()
        print(f"✓ Wrote repaired dataset to {output_file.name}")
        print(f"✓ Wrote repair log to {log_file.name}")
        print()
        print("=" * 70)
        print("✓ Phase 3 repair complete!")
        print("=" * 70)
        print("\nNext step: Re-run Phase 2 validation on repaired dataset")
        print("  python3 scripts/2_validate_dataset.py --input outputs/phase3_repair/repaired_dataset.csv")

    return {
        'input_file': input_file,
        'error_file': error_file,
        'repaired_csv': output_file,
        'log_file': log_file,
        'rows_repaired': rows_repaired,
        'total_actions': total_actions,
        'schema_hash': schema_hash,
    }


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_file = project_root / 'outputs' / 'phase1_extraction' / 'main_dataset.csv'
    error_file = project_root / 'outputs' / 'phase2_validation' / 'validation_errors.csv'
    run_phase3(input_file, error_file, verbose=True)


if __name__ == '__main__':
    main()
