#!/usr/bin/env python3
"""
Phase 1: Parse AI Responses to CSV

Parses AI-responses.txt (delimited format with Answer N: VALUE lines) into:
- main_dataset.csv: Valid responses with all 77 answers
- data_with_errors.csv: Malformed responses for manual review

Input:  /raw_data/AI_analysis/AI-responses.txt
Output: /outputs/phase1_extraction/main_dataset.csv
        /outputs/phase1_extraction/data_with_errors.csv
        /outputs/phase1_extraction/extraction_log.txt
"""

import re
import csv
import os
from pathlib import Path
from datetime import datetime

# Field mapping: Answer number → CSV column name (per schema)
FIELD_MAPPING = {
    1: 'a1_country_code',
    2: 'a2_authority_name',
    3: 'a3_appellate_decision',
    4: 'a4_decision_year',
    5: 'a5_decision_month',
    6: 'a6_num_defendants',
    7: 'a7_defendant_name',
    8: 'a8_defendant_class',
    9: 'a9_enterprise_size',
    10: 'a10_gov_level',
    11: 'a11_defendant_role',
    12: 'a12_sector',
    13: 'a13_sector_other',
    14: 'a14_processing_contexts',
    15: 'a15_data_subject_complaint',
    16: 'a16_media_attention',
    17: 'a17_official_audit',
    18: 'a18_art33_discussed',
    19: 'a19_art33_breached',
    20: 'a20_breach_notification_effect',
    21: 'a21_art5_lawfulness_fairness',
    22: 'a22_art5_purpose_limitation',
    23: 'a23_art5_data_minimization',
    24: 'a24_art5_accuracy',
    25: 'a25_art5_storage_limitation',
    26: 'a26_art5_integrity_confidentiality',
    27: 'a27_art5_accountability',
    28: 'a28_art9_discussed',
    29: 'a29_vulnerable_groups',
    30: 'a30_art6_consent',
    31: 'a31_art6_contract',
    32: 'a32_art6_legal_obligation',
    33: 'a33_art6_vital_interests',
    34: 'a34_art6_public_task',
    35: 'a35_art6_legitimate_interest',
    36: 'a36_legal_basis_summary',
    37: 'a37_right_access_violated',
    38: 'a38_right_rectification_violated',
    39: 'a39_right_erasure_violated',
    40: 'a40_right_restriction_violated',
    41: 'a41_right_portability_violated',
    42: 'a42_right_object_violated',
    43: 'a43_transparency_violated',
    44: 'a44_automated_decisions_violated',
    45: 'a45_warning_issued',
    46: 'a46_reprimand_issued',
    47: 'a47_comply_data_subject_order',
    48: 'a48_compliance_order',
    49: 'a49_breach_communication_order',
    50: 'a50_erasure_restriction_order',
    51: 'a51_certification_withdrawal',
    52: 'a52_data_flow_suspension',
    53: 'a53_fine_imposed',
    54: 'a54_fine_amount',
    55: 'a55_fine_currency',
    56: 'a56_turnover_discussed',
    57: 'a57_turnover_amount',
    58: 'a58_turnover_currency',
    59: 'a59_nature_gravity_duration',
    60: 'a60_intentional_negligent',
    61: 'a61_mitigate_damage_actions',
    62: 'a62_technical_org_measures',
    63: 'a63_previous_infringements',
    64: 'a64_cooperation_authority',
    65: 'a65_data_categories_affected',
    66: 'a66_infringement_became_known',
    67: 'a67_prior_orders_compliance',
    68: 'a68_codes_certification',
    69: 'a69_other_factors',
    70: 'a70_systematic_art83_discussion',
    71: 'a71_first_violation',
    72: 'a72_cross_border_oss',
    73: 'a73_oss_role',
    74: 'a74_guidelines_referenced',
    75: 'a75_case_summary',
    76: 'a76_art83_weighing_summary',
    77: 'a77_articles_breached'
}

# CSV header: ID column + all 77 schema fields
CSV_HEADERS = ['id'] + [FIELD_MAPPING[i] for i in range(1, 78)]


def parse_response_file(filepath: str) -> tuple[list[dict], list[dict]]:
    """
    Parse AI-responses.txt into valid and malformed response dicts.

    Returns:
        (valid_responses, malformed_responses)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex pattern for response blocks:
    # Single delimiter → ID → delimiter → answers → delimiter
    pattern = (
        r'----- RESPONSE DELIMITER -----\s*\n'
        r'ID: ([^\n]+)\s*\n'
        r'----- RESPONSE DELIMITER -----\s*\n'
        r'(.*?)'
        r'----- RESPONSE DELIMITER -----'
    )

    matches = re.findall(pattern, content, re.DOTALL)

    valid_responses = []
    malformed_responses = []

    for id_value, answer_block in matches:
        id_clean = id_value.strip()

        # Extract all "Answer N: VALUE" lines
        answer_lines = re.findall(r'^Answer (\d+): (.*)$', answer_block, re.MULTILINE)

        # Build answer dict
        answers = {}
        for num_str, value in answer_lines:
            num = int(num_str)
            # Strip leading/trailing whitespace
            cleaned_value = value.strip()
            answers[num] = cleaned_value

        # Validate: must have exactly answers 1-77
        answer_nums = sorted(answers.keys())
        expected_nums = list(range(1, 78))

        if answer_nums == expected_nums:
            # Valid response
            row = {'id': id_clean}
            for num in range(1, 78):
                field_name = FIELD_MAPPING[num]
                row[field_name] = answers[num]
            valid_responses.append(row)
        else:
            # Malformed response
            row = {
                'id': id_clean,
                'error': f"Expected 77 answers (1-77), got {len(answers)} answers: {answer_nums}",
                'raw_answers': str(answers)
            }
            malformed_responses.append(row)

    return valid_responses, malformed_responses


def write_csv(filepath: str, rows: list[dict], fieldnames: list[str]):
    """Write rows to CSV with proper quoting."""
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


def main():
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_file = project_root / 'raw_data' / 'AI_analysis' / 'AI-responses.txt'
    output_main = project_root / 'outputs' / 'phase1_extraction' / 'main_dataset.csv'
    output_errors = project_root / 'outputs' / 'phase1_extraction' / 'data_with_errors.csv'
    output_log = project_root / 'outputs' / 'phase1_extraction' / 'extraction_log.txt'

    # Ensure output directory exists
    output_main.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1: Parsing AI Responses to CSV")
    print("=" * 60)
    print(f"Input:  {input_file}")
    print(f"Output: {output_main}")
    print(f"Errors: {output_errors}")
    print(f"Log:    {output_log}")
    print()

    # Parse responses
    print("Parsing responses...")
    valid_responses, malformed_responses = parse_response_file(str(input_file))

    print(f"✓ Valid responses:     {len(valid_responses)}")
    print(f"✗ Malformed responses: {len(malformed_responses)}")
    print()

    # Write valid responses
    if valid_responses:
        print(f"Writing {len(valid_responses)} valid rows to {output_main.name}...")
        write_csv(str(output_main), valid_responses, CSV_HEADERS)
        print(f"✓ Wrote {output_main}")
    else:
        print("⚠ No valid responses to write!")

    # Write malformed responses
    if malformed_responses:
        print(f"\nWriting {len(malformed_responses)} malformed rows to {output_errors.name}...")
        error_headers = ['id', 'error', 'raw_answers']
        write_csv(str(output_errors), malformed_responses, error_headers)
        print(f"✓ Wrote {output_errors}")
        print("\n⚠ Review malformed responses manually:")
        for row in malformed_responses:
            print(f"  - {row['id']}: {row['error']}")
    else:
        print("\n✓ No malformed responses")

    # Write extraction log
    print(f"\nWriting extraction log to {output_log.name}...")
    with open(str(output_log), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Phase 1: AI Response Extraction Log\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Output directory: {output_main.parent}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Total responses found: {len(valid_responses) + len(malformed_responses)}\n")
        f.write(f"  Valid responses (77 answers): {len(valid_responses)}\n")
        f.write(f"  Malformed responses: {len(malformed_responses)}\n\n")
        if malformed_responses:
            f.write("Malformed response details:\n")
            for row in malformed_responses:
                f.write(f"  - {row['id']}: {row['error']}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Extraction complete. Ready for Phase 2 validation.\n")
        f.write("=" * 60 + "\n")
    print(f"✓ Wrote {output_log}")

    print()
    print("=" * 60)
    print("✓ Phase 1 extraction complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
