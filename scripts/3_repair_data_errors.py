#!/usr/bin/env python3
"""
Phase 3: Auto-Repair Common Data Errors

Automatically fixes targeted validation errors identified in Phase 2:
1. NOT_DISCUSSED → NO (two-option YES/NO fields)
2. NOT_APPLICABLE → NOT_DISCUSSED (three-option fields with NOT_DISCUSSED sentinel)

Input:  /outputs/phase1_extraction/main_dataset.csv
        /outputs/phase2_validation/validation_errors.csv
Output: /outputs/phase3_repair/repaired_dataset.csv
        /outputs/phase3_repair/repair_log.txt
"""

import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, Optional

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
        'reason': 'Two-option field expects YES or NO, NOT_DISCUSSED invalid'
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

    # Pattern 5: Sector values in defendant_class → OTHER
    'pattern5_sector_in_defendant_class': {
        'fields': ['a8_defendant_class'],
        'from_values': ['MEDIA', 'EDUCATION', 'JUDICIAL', 'TELECOM'],
        'to_value': 'OTHER',
        'reason': 'Sector values incorrectly used in defendant_class field'
    },

    # Pattern 6: INTENTIONAL → AGGRAVATING (Art 83 factors)
    'pattern6_intentional_to_aggravating': {
        'fields': ['a60_intentional_negligent'],
        'from_value': 'INTENTIONAL',
        'to_value': 'AGGRAVATING',
        'reason': 'Field expects AGGRAVATING/MITIGATING/NEUTRAL/NOT_DISCUSSED, not INTENTIONAL'
    }
}


def load_validation_errors(error_file: str) -> dict:
    """Load validation errors and group by row_id and field."""
    errors_by_row = defaultdict(lambda: defaultdict(list))

    with open(error_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for error in reader:
            if error['severity'] == 'ERROR':  # Only fix ERRORS, not WARNINGS
                row_id = error['row_id']
                field = error['field_name']
                errors_by_row[row_id][field].append(error)

    return errors_by_row


def apply_repairs(row: dict, row_id: str, errors_by_row: dict) -> dict:
    """
    Apply automatic repairs to a row.
    Returns (repaired_row, repairs_applied)
    """
    repaired = row.copy()
    repairs = []

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
                    repairs.append({
                        'field': field,
                        'from': rule['from_value'],
                        'to': rule['to_value'],
                        'pattern': pattern_name,
                        'reason': rule['reason']
                    })

            # Handle rules with multiple from_values (e.g., pattern5)
            elif 'from_values' in rule:
                if repaired[field] in rule['from_values'] and field in row_errors:
                    old_value = repaired[field]
                    repaired[field] = rule['to_value']
                    repairs.append({
                        'field': field,
                        'from': old_value,
                        'to': rule['to_value'],
                        'pattern': pattern_name,
                        'reason': rule['reason']
                    })

    return repaired, repairs


def repair_dataset(input_file: str, error_file: str, output_file: str, log_file: str):
    """Repair dataset and generate log."""

    # Load validation errors
    print("Loading validation errors...")
    errors_by_row = load_validation_errors(error_file)
    print(f"✓ Loaded errors for {len(errors_by_row)} rows")

    # Load and repair dataset
    print(f"\nRepairing dataset...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    repaired_rows = []
    all_repairs = defaultdict(int)
    rows_repaired = 0
    repair_details = []

    for row in rows:
        row_id = row.get('id', 'unknown')
        repaired_row, repairs = apply_repairs(row, row_id, errors_by_row)

        if repairs:
            rows_repaired += 1
            for repair in repairs:
                all_repairs[repair['pattern']] += 1
                repair_details.append({
                    'row_id': row_id,
                    'field': repair['field'],
                    'from_value': repair['from'],
                    'to_value': repair['to'],
                    'pattern': repair['pattern']
                })

        repaired_rows.append(repaired_row)

    # Write repaired dataset
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(repaired_rows)

    print(f"✓ Repaired {rows_repaired} rows")
    print(f"✓ Total repairs: {sum(all_repairs.values())}")

    # Generate repair log
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Phase 3: Data Repair Log\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Input: {input_file}\n")
        f.write(f"Output: {output_file}\n\n")

        f.write("REPAIR SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total rows processed: {len(rows)}\n")
        f.write(f"Rows repaired: {rows_repaired}\n")
        f.write(f"Total repairs applied: {sum(all_repairs.values())}\n\n")

        f.write("REPAIRS BY PATTERN\n")
        f.write("-" * 70 + "\n")
        for pattern, count in sorted(all_repairs.items()):
            f.write(f"{pattern}: {count} repairs\n")
            rule = REPAIR_RULES[pattern]
            if 'from_value' in rule:
                f.write(f"  Rule: {rule['from_value']} → {rule['to_value']}\n")
            elif 'from_values' in rule:
                from_vals = ', '.join(rule['from_values'])
                f.write(f"  Rule: [{from_vals}] → {rule['to_value']}\n")
            f.write(f"  Reason: {rule['reason']}\n\n")

        f.write("DETAILED REPAIRS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Row ID, Field, From, To, Pattern\n")
        for detail in repair_details[:100]:  # First 100 for brevity
            f.write(f"{detail['row_id']}, {detail['field']}, {detail['from_value']}, {detail['to_value']}, {detail['pattern']}\n")

        if len(repair_details) > 100:
            f.write(f"... and {len(repair_details) - 100} more repairs\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Repair complete. Run Phase 2 validation again to verify.\n")
        f.write("=" * 70 + "\n")

    return rows_repaired, sum(all_repairs.values())


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

    rows_repaired, total_repairs = repair_dataset(
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
        'total_repairs': total_repairs,
    }


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_file = project_root / 'outputs' / 'phase1_extraction' / 'main_dataset.csv'
    error_file = project_root / 'outputs' / 'phase2_validation' / 'validation_errors.csv'
    run_phase3(input_file, error_file, verbose=True)


if __name__ == '__main__':
    main()
