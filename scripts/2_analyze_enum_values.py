#!/usr/bin/env python3
"""
Phase 2 Utility: Enum Value Frequency Analysis

Analyzes all enum fields in the dataset to show:
- Actual values present in data
- Frequency counts
- Schema compliance status (✓ valid, ✗ invalid)

This helps identify common mismappings for potential repair rules.

Input:  /outputs/phase1_extraction/main_dataset.csv (or custom path)
Output: /outputs/phase2_validation/enum_analysis.txt
        /outputs/phase2_validation/enum_analysis.csv
"""

import csv
from pathlib import Path
from collections import defaultdict, Counter
import sys

# Import validation rules from main validator
# We'll define them here for independence
ENUM_FIELDS = {
    # Section 1: Basic Case Metadata
    'a1_country_code': ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI',
                       'FR', 'GB', 'GR', 'HU', 'IE', 'IS', 'IT', 'LI', 'LT', 'LU', 'LV', 'MT',
                       'NL', 'NO', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK'],
    'a3_appellate_decision': ['YES', 'NO', 'NOT_DISCUSSED'],

    # Section 2: Defendant Information
    'a8_defendant_class': ['PUBLIC', 'PRIVATE', 'NGO_OR_NONPROFIT', 'RELIGIOUS', 'INDIVIDUAL', 'POLITICAL_PARTY', 'OTHER'],
    'a9_enterprise_size': ['SME', 'LARGE', 'VERY_LARGE', 'UNKNOWN', 'NOT_APPLICABLE'],
    'a10_gov_level': ['STATE', 'MUNICIPAL', 'JUDICIAL', 'UNKNOWN', 'NOT_APPLICABLE'],
    'a11_defendant_role': ['CONTROLLER', 'PROCESSOR', 'JOINT_CONTROLLER', 'NOT_MENTIONED'],
    'a12_sector': ['HEALTH', 'EDUCATION', 'TELECOM', 'RETAIL', 'DIGITAL_SERVICES', 'FINANCIAL',
                  'CONSULTING', 'HOUSING_TOURISM', 'FITNESS_WELLNESS', 'MEDIA', 'MANUFACTURING',
                  'UTILITIES', 'MEMBERSHIP_ORGS', 'RELIGIOUS_ORGS', 'TAX', 'OTHER_PUBLIC_ADMIN', 'OTHER'],

    # Section 3-4: Processing Context and Case Origins
    'a15_data_subject_complaint': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a16_media_attention': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a17_official_audit': ['YES', 'NO', 'NOT_DISCUSSED'],

    # Section 5: Article 33
    'a18_art33_discussed': ['YES', 'NO'],
    'a19_art33_breached': ['YES', 'NO', 'NOT_DISCUSSED', 'NOT_APPLICABLE'],
    'a20_breach_notification_effect': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED'],

    # Section 6: Article 5 Principles
    'a21_art5_lawfulness_fairness': ['BREACHED', 'NOT_BREACHED', 'NOT_DISCUSSED'],
    'a22_art5_purpose_limitation': ['BREACHED', 'NOT_BREACHED', 'NOT_DISCUSSED'],
    'a23_art5_data_minimization': ['BREACHED', 'NOT_BREACHED', 'NOT_DISCUSSED'],
    'a24_art5_accuracy': ['BREACHED', 'NOT_BREACHED', 'NOT_DISCUSSED'],
    'a25_art5_storage_limitation': ['BREACHED', 'NOT_BREACHED', 'NOT_DISCUSSED'],
    'a26_art5_integrity_confidentiality': ['BREACHED', 'NOT_BREACHED', 'NOT_DISCUSSED'],
    'a27_art5_accountability': ['BREACHED', 'NOT_BREACHED', 'NOT_DISCUSSED'],

    # Section 7: Special Categories
    'a28_art9_discussed': ['YES', 'NO'],

    # Section 8: Article 6 Legal Bases
    'a30_art6_consent': ['VALID', 'INVALID', 'NOT_DISCUSSED'],
    'a31_art6_contract': ['VALID', 'INVALID', 'NOT_DISCUSSED'],
    'a32_art6_legal_obligation': ['VALID', 'INVALID', 'NOT_DISCUSSED'],
    'a33_art6_vital_interests': ['VALID', 'INVALID', 'NOT_DISCUSSED'],
    'a34_art6_public_task': ['VALID', 'INVALID', 'NOT_DISCUSSED'],
    'a35_art6_legitimate_interest': ['VALID', 'INVALID', 'NOT_DISCUSSED'],

    # Section 9: Data Subject Rights
    'a37_right_access_violated': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a38_right_rectification_violated': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a39_right_erasure_violated': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a40_right_restriction_violated': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a41_right_portability_violated': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a42_right_object_violated': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a43_transparency_violated': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a44_automated_decisions_violated': ['YES', 'NO', 'NOT_DISCUSSED'],

    # Section 10: Article 58 Corrective Measures
    'a45_warning_issued': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a46_reprimand_issued': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a47_comply_data_subject_order': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a48_compliance_order': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a49_breach_communication_order': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a50_erasure_restriction_order': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a51_certification_withdrawal': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a52_data_flow_suspension': ['YES', 'NO', 'NOT_DISCUSSED'],

    # Section 11: Financial Penalties
    'a53_fine_imposed': ['YES', 'NO'],
    'a55_fine_currency': ['EUR', 'GBP', 'SEK', 'DKK', 'NOK', 'PLN', 'CZK', 'HUF', 'RON', 'BGN', 'HRK', 'ISK', 'NOT_APPLICABLE'],
    'a56_turnover_discussed': ['YES', 'NO'],
    'a58_turnover_currency': ['EUR', 'GBP', 'SEK', 'DKK', 'NOK', 'PLN', 'CZK', 'HUF', 'RON', 'BGN', 'HRK', 'ISK',
                             'NOT_APPLICABLE', 'NOT_DISCUSSED'],

    # Section 12: Article 83(2) Factors
    'a59_nature_gravity_duration': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED'],
    'a60_intentional_negligent': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED'],
    'a61_mitigate_damage_actions': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED'],
    'a62_technical_org_measures': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED'],
    'a63_previous_infringements': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED'],
    'a64_cooperation_authority': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED'],
    'a65_data_categories_affected': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED'],
    'a66_infringement_became_known': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED'],
    'a67_prior_orders_compliance': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED'],
    'a68_codes_certification': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED'],
    'a69_other_factors': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED'],
    'a70_systematic_art83_discussion': ['YES', 'NO'],
    'a71_first_violation': ['YES_FIRST_TIME', 'NO_REPEAT_OFFENDER', 'NOT_DISCUSSED'],

    # Section 13: Cross-Border & References
    'a72_cross_border_oss': ['YES', 'NO', 'NOT_DISCUSSED'],
    'a73_oss_role': ['LEAD', 'CONCERNED', 'NOT_APPLICABLE'],
}


def analyze_enum_values(input_file: str) -> dict:
    """
    Analyze all enum fields and return value frequencies.
    Returns: {field_name: Counter({value: count, ...})}
    """
    # Read dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Analyze each enum field
    field_values = {}
    for field_name in ENUM_FIELDS.keys():
        if field_name in rows[0]:  # Field exists in data
            values = [row[field_name] for row in rows]
            field_values[field_name] = Counter(values)

    return field_values


def generate_text_report(field_values: dict, output_file: str):
    """Generate human-readable text report."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ENUM VALUE FREQUENCY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Legend: ✓ = Valid (in schema), ✗ = Invalid (not in schema)\n\n")

        for field_name in sorted(ENUM_FIELDS.keys()):
            if field_name not in field_values:
                continue

            allowed_values = set(ENUM_FIELDS[field_name])
            actual_values = field_values[field_name]

            f.write("-" * 80 + "\n")
            f.write(f"{field_name}\n")
            f.write(f"Schema allows: {', '.join(sorted(allowed_values))}\n\n")

            # Separate valid and invalid values
            valid_values = [(v, c) for v, c in actual_values.items() if v in allowed_values]
            invalid_values = [(v, c) for v, c in actual_values.items() if v not in allowed_values]

            # Show valid values
            if valid_values:
                f.write("Valid values:\n")
                for value, count in sorted(valid_values, key=lambda x: x[1], reverse=True):
                    f.write(f"  ✓ {value:30s} {count:5d}\n")

            # Show invalid values (highlight these!)
            if invalid_values:
                f.write("\nInvalid values (NOT IN SCHEMA):\n")
                for value, count in sorted(invalid_values, key=lambda x: x[1], reverse=True):
                    f.write(f"  ✗ {value:30s} {count:5d}  ← FIX NEEDED\n")

            # Summary
            total = sum(actual_values.values())
            invalid_count = sum(c for v, c in invalid_values)
            if invalid_count > 0:
                f.write(f"\nSummary: {invalid_count}/{total} ({invalid_count/total*100:.1f}%) values are invalid\n")

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


def generate_csv_report(field_values: dict, output_file: str):
    """Generate CSV report for easy filtering/sorting."""
    rows = []

    for field_name in sorted(ENUM_FIELDS.keys()):
        if field_name not in field_values:
            continue

        allowed_values = set(ENUM_FIELDS[field_name])
        actual_values = field_values[field_name]

        for value, count in sorted(actual_values.items(), key=lambda x: x[1], reverse=True):
            is_valid = value in allowed_values
            rows.append({
                'field_name': field_name,
                'value': value,
                'count': count,
                'valid': 'YES' if is_valid else 'NO',
                'status': '✓' if is_valid else '✗ FIX'
            })

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['field_name', 'value', 'count', 'valid', 'status'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Check for custom input
    if len(sys.argv) > 1 and sys.argv[1] == '--input':
        input_file = Path(sys.argv[2]) if not sys.argv[2].startswith('/') else Path(sys.argv[2])
        if not input_file.is_absolute():
            input_file = project_root / input_file
        output_dir = input_file.parent
    else:
        input_file = project_root / 'outputs' / 'phase1_extraction' / 'main_dataset.csv'
        output_dir = project_root / 'outputs' / 'phase2_validation'

    output_txt = output_dir / 'enum_analysis.txt'
    output_csv = output_dir / 'enum_analysis.csv'

    print("=" * 80)
    print("Enum Value Frequency Analysis")
    print("=" * 80)
    print(f"Input:  {input_file}")
    print(f"Output: {output_txt}")
    print(f"        {output_csv}")
    print()

    # Analyze
    print("Analyzing enum fields...")
    field_values = analyze_enum_values(str(input_file))
    print(f"✓ Analyzed {len(field_values)} enum fields")

    # Count total invalid values
    total_invalid = 0
    invalid_fields = 0
    for field_name, values in field_values.items():
        allowed = set(ENUM_FIELDS[field_name])
        invalid = sum(count for value, count in values.items() if value not in allowed)
        if invalid > 0:
            total_invalid += invalid
            invalid_fields += 1

    print(f"✓ Found {total_invalid} invalid values across {invalid_fields} fields")

    # Generate reports
    print("\nGenerating reports...")
    generate_text_report(field_values, str(output_txt))
    generate_csv_report(field_values, str(output_csv))

    print(f"✓ Wrote {output_txt.name}")
    print(f"✓ Wrote {output_csv.name}")

    print()
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review enum_analysis.txt for invalid values")
    print("  2. Identify common mismappings (e.g., BREACHED used for violated field)")
    print("  3. Add repair rules to scripts/3_repair_data_errors.py")
    print("  4. Re-run repair and validation pipeline")


if __name__ == '__main__':
    main()
