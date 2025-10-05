#!/usr/bin/env python3
"""
Phase 2: Comprehensive CSV Validation

Validates EVERY field in main_dataset.csv against schema rules.
Outputs validated data and detailed error reports.

Input:  /outputs/phase1_extraction/main_dataset.csv
Output: /outputs/phase2_validation/validated_data.csv
        /outputs/phase2_validation/validation_errors.csv
        /outputs/phase2_validation/validation_report.txt

When run with `--input <path/to/dataset.csv>`, the validator writes
`<dataset>_validated.csv`, `<dataset>_validation_errors.csv`, and
`<dataset>_validation_report.txt` next to the provided file to avoid
overwriting phase artefacts.
"""

import csv
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, Optional

from schema_utils import compute_schema_hash, resolve_schema_files, write_schema_snapshot

# ============================================================================
# VALIDATION RULES FOR ALL 77 FIELDS
# ============================================================================

VALIDATION_RULES = {
    # Section 1: Basic Case Metadata
    'a1_country_code': {
        'type': 'enum',
        'allowed': ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI',
                   'FR', 'GB', 'GR', 'HU', 'IE', 'IS', 'IT', 'LI', 'LT', 'LU', 'LV', 'MT',
                   'NL', 'NO', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
    },
    'a2_authority_name': {
        'type': 'free_text',
        'non_empty': True,
        'max_length': 255,
    },
    'a3_appellate_decision': {
        'type': 'enum',
        'allowed': ['YES', 'NO', 'NOT_DISCUSSED']
    },
    'a4_decision_year': {
        'type': 'integer_range',
        'min': 2018,
        'max': 2025,
        'warn_outside': True,
        'future_tolerance_years': 2,
    },
    'a5_decision_month': {
        'type': 'integer_range',
        'min': 0,
        'max': 12
    },

    # Section 2: Defendant Information
    'a6_num_defendants': {
        'type': 'integer_range',
        'min': 1,
        'max': 999
    },
    'a7_defendant_name': {
        'type': 'free_text',
        'non_empty': True,
        'max_length': 255,
    },
    'a8_defendant_class': {
        'type': 'enum',
        'allowed': ['PUBLIC', 'PRIVATE', 'NGO_OR_NONPROFIT', 'RELIGIOUS', 'INDIVIDUAL', 'POLITICAL_PARTY', 'OTHER']
    },
    'a9_enterprise_size': {
        'type': 'enum',
        'allowed': ['SME', 'LARGE', 'VERY_LARGE', 'UNKNOWN', 'NOT_APPLICABLE']
    },
    'a10_gov_level': {
        'type': 'enum',
        'allowed': ['STATE', 'MUNICIPAL', 'JUDICIAL', 'UNKNOWN', 'NOT_APPLICABLE']
    },
    'a11_defendant_role': {
        'type': 'enum',
        'allowed': ['CONTROLLER', 'PROCESSOR', 'JOINT_CONTROLLER', 'NOT_MENTIONED']
    },
    'a12_sector': {
        'type': 'enum',
        'allowed': ['HEALTH', 'EDUCATION', 'TELECOM', 'RETAIL', 'DIGITAL_SERVICES', 'FINANCIAL',
                   'CONSULTING', 'HOUSING_TOURISM', 'FITNESS_WELLNESS', 'MEDIA', 'MANUFACTURING',
                   'UTILITIES', 'MEMBERSHIP_ORGS', 'RELIGIOUS_ORGS', 'TAX', 'OTHER_PUBLIC_ADMIN', 'OTHER']
    },
    'a13_sector_other': {
        'type': 'free_text_or_sentinel',
        'sentinel': 'NOT_APPLICABLE',
        'max_length': 255,
    },

    # Section 3-4: Processing Context and Case Origins
    'a14_processing_contexts': {
        'type': 'semicolon_list_or_sentinel',
        'allowed_tokens': ['CCTV', 'MARKETING', 'RECRUITMENT_AND_HR', 'COOKIES', 'CUSTOMER_LOYALTY_CLUBS',
                          'CREDIT_SCORING', 'BACKGROUND_CHECKS', 'ARTIFICIAL_INTELLIGENCE',
                          'PROBLEMATIC_THIRD_PARTY_SHARING_STATED', 'DPO_ROLE_PROBLEMS_STATED',
                          'JOURNALISM', 'EMPLOYEE_MONITORING'],
        'sentinel': 'NOT_DISCUSSED'
    },
    'a15_data_subject_complaint': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a16_media_attention': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a17_official_audit': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},

    # Section 5: Article 33
    'a18_art33_discussed': {'type': 'enum', 'allowed': ['YES', 'NO']},
    'a19_art33_breached': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED', 'NOT_APPLICABLE']},
    'a20_breach_notification_effect': {'type': 'enum', 'allowed': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED']},

    # Section 6: Article 5 Principles (a21-a27)
    'a21_art5_lawfulness_fairness': {'type': 'enum', 'allowed': ['BREACHED', 'NOT_BREACHED', 'NOT_DISCUSSED']},
    'a22_art5_purpose_limitation': {'type': 'enum', 'allowed': ['BREACHED', 'NOT_BREACHED', 'NOT_DISCUSSED']},
    'a23_art5_data_minimization': {'type': 'enum', 'allowed': ['BREACHED', 'NOT_BREACHED', 'NOT_DISCUSSED']},
    'a24_art5_accuracy': {'type': 'enum', 'allowed': ['BREACHED', 'NOT_BREACHED', 'NOT_DISCUSSED']},
    'a25_art5_storage_limitation': {'type': 'enum', 'allowed': ['BREACHED', 'NOT_BREACHED', 'NOT_DISCUSSED']},
    'a26_art5_integrity_confidentiality': {'type': 'enum', 'allowed': ['BREACHED', 'NOT_BREACHED', 'NOT_DISCUSSED']},
    'a27_art5_accountability': {'type': 'enum', 'allowed': ['BREACHED', 'NOT_BREACHED', 'NOT_DISCUSSED']},

    # Section 7: Special Categories
    'a28_art9_discussed': {'type': 'enum', 'allowed': ['YES', 'NO']},
    'a29_vulnerable_groups': {
        'type': 'semicolon_list_or_sentinel',
        'allowed_tokens': ['CHILDREN', 'EMPLOYEES', 'ELDERLY', 'PATIENTS', 'STUDENTS', 'OTHER_VULNERABLE'],
        'sentinel': 'NOT_DISCUSSED'
    },

    # Section 8: Article 6 Legal Bases (a30-a35)
    'a30_art6_consent': {'type': 'enum', 'allowed': ['VALID', 'INVALID', 'NOT_DISCUSSED']},
    'a31_art6_contract': {'type': 'enum', 'allowed': ['VALID', 'INVALID', 'NOT_DISCUSSED']},
    'a32_art6_legal_obligation': {'type': 'enum', 'allowed': ['VALID', 'INVALID', 'NOT_DISCUSSED']},
    'a33_art6_vital_interests': {'type': 'enum', 'allowed': ['VALID', 'INVALID', 'NOT_DISCUSSED']},
    'a34_art6_public_task': {'type': 'enum', 'allowed': ['VALID', 'INVALID', 'NOT_DISCUSSED']},
    'a35_art6_legitimate_interest': {'type': 'enum', 'allowed': ['VALID', 'INVALID', 'NOT_DISCUSSED']},
    'a36_legal_basis_summary': {
        'type': 'free_text_or_sentinel',
        'sentinel': 'NOT_DISCUSSED',
        'max_length': 2000,
    },

    # Section 9: Data Subject Rights (a37-a44)
    'a37_right_access_violated': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a38_right_rectification_violated': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a39_right_erasure_violated': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a40_right_restriction_violated': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a41_right_portability_violated': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a42_right_object_violated': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a43_transparency_violated': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a44_automated_decisions_violated': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},

    # Section 10: Article 58 Corrective Measures (a45-a52)
    'a45_warning_issued': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a46_reprimand_issued': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a47_comply_data_subject_order': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a48_compliance_order': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a49_breach_communication_order': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a50_erasure_restriction_order': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a51_certification_withdrawal': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a52_data_flow_suspension': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},

    # Section 11: Financial Penalties
    'a53_fine_imposed': {'type': 'enum', 'allowed': ['YES', 'NO']},
    'a54_fine_amount': {'type': 'integer_or_sentinel', 'sentinel': 'NOT_APPLICABLE'},
    'a55_fine_currency': {
        'type': 'enum',
        'allowed': ['EUR', 'GBP', 'SEK', 'DKK', 'NOK', 'PLN', 'CZK', 'HUF', 'RON', 'BGN', 'HRK', 'CHF', 'ISK', 'USD', 'NOT_APPLICABLE']
    },
    'a56_turnover_discussed': {'type': 'enum', 'allowed': ['YES', 'NO']},
    'a57_turnover_amount': {'type': 'integer_or_sentinels', 'sentinels': ['NOT_APPLICABLE', 'NOT_DISCUSSED']},
    'a58_turnover_currency': {
        'type': 'enum',
        'allowed': ['EUR', 'GBP', 'SEK', 'DKK', 'NOK', 'PLN', 'CZK', 'HUF', 'RON', 'BGN', 'HRK', 'CHF', 'ISK', 'USD',
                   'NOT_APPLICABLE', 'NOT_DISCUSSED']
    },

    # Section 12: Article 83(2) Factors (a59-a69)
    'a59_nature_gravity_duration': {'type': 'enum', 'allowed': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED']},
    'a60_intentional_negligent': {'type': 'enum', 'allowed': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED']},
    'a61_mitigate_damage_actions': {'type': 'enum', 'allowed': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED']},
    'a62_technical_org_measures': {'type': 'enum', 'allowed': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED']},
    'a63_previous_infringements': {'type': 'enum', 'allowed': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED']},
    'a64_cooperation_authority': {'type': 'enum', 'allowed': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED']},
    'a65_data_categories_affected': {'type': 'enum', 'allowed': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED']},
    'a66_infringement_became_known': {'type': 'enum', 'allowed': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED']},
    'a67_prior_orders_compliance': {'type': 'enum', 'allowed': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED']},
    'a68_codes_certification': {'type': 'enum', 'allowed': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED']},
    'a69_other_factors': {'type': 'enum', 'allowed': ['AGGRAVATING', 'MITIGATING', 'NEUTRAL', 'NOT_DISCUSSED']},
    'a70_systematic_art83_discussion': {'type': 'enum', 'allowed': ['YES', 'NO']},
    'a71_first_violation': {'type': 'enum', 'allowed': ['YES_FIRST_TIME', 'NO_REPEAT_OFFENDER', 'NOT_DISCUSSED']},

    # Section 13: Cross-Border & References
    'a72_cross_border_oss': {'type': 'enum', 'allowed': ['YES', 'NO', 'NOT_DISCUSSED']},
    'a73_oss_role': {'type': 'enum', 'allowed': ['LEAD', 'CONCERNED', 'NOT_APPLICABLE']},
    'a74_guidelines_referenced': {
        'type': 'free_text_or_sentinel',
        'sentinel': 'NOT_APPLICABLE',
        'max_length': 1000,
    },

    # Section 14: Summaries
    'a75_case_summary': {
        'type': 'free_text',
        'non_empty': True,
        'max_length': 4000,
    },
    'a76_art83_weighing_summary': {
        'type': 'free_text_or_sentinel',
        'sentinel': 'NOT_DISCUSSED',
        'max_length': 4000,
    },
    'a77_articles_breached': {
        'type': 'semicolon_integers_or_sentinels',
        'sentinels': ['NONE_VIOLATED', 'NOT_DISCUSSED']
    }
}

BINARY_ONLY_FIELDS = {
    'a18_art33_discussed',
    'a28_art9_discussed',
    'a53_fine_imposed',
    'a56_turnover_discussed',
    'a70_systematic_art83_discussion',
}

FORBIDDEN_BINARY_SENTINELS = {'NOT_APPLICABLE', 'NOT_DISCUSSED'}

ARTICLE_PREFIX_PATTERN = re.compile(r'^(art\.?|article)\s*', re.IGNORECASE)
GDPR_SUFFIX_PATTERN = re.compile(r'(gdpr)$', re.IGNORECASE)

DEFAULT_FREE_TEXT_PATTERNS = [
    re.compile(r'lorem ipsum', re.IGNORECASE),
    re.compile(r'placeholder text', re.IGNORECASE),
]


def normalize_article_token(token: str) -> str:
    """Normalize GDPR article token to its leading numeric anchor."""
    stripped = token.strip()
    if not stripped:
        return ''

    stripped = ARTICLE_PREFIX_PATTERN.sub('', stripped)
    stripped = GDPR_SUFFIX_PATTERN.sub('', stripped)
    stripped = stripped.replace(' ', '')

    match = re.match(r'(\d+)', stripped)
    if match:
        return match.group(1)

    match = re.search(r'(\d+)', stripped)
    if match:
        return match.group(1)

    return stripped


def parse_article_numbers(value: str) -> set[str]:
    """Parse semicolon-separated article list into a set of numeric strings."""
    if not value or value in {'NONE_VIOLATED', 'NOT_DISCUSSED'}:
        return set()

    tokens = [normalize_article_token(tok) for tok in value.split(';')]
    parsed_tokens: set[str] = set()
    for token in tokens:
        ok, number, _ = parse_int(token)
        if ok and number is not None:
            parsed_tokens.add(str(number))
    return parsed_tokens


def parse_int(value: str, *, allow_negative: bool = False) -> tuple[bool, Optional[int], str]:
    """Attempt to parse an integer, flagging empty or negative placeholders."""
    normalized = value.strip()
    if normalized == '':
        return False, None, 'Empty value; expected integer'
    try:
        number = int(normalized)
    except ValueError:
        return False, None, f"Expected integer but received '{value}'"
    if not allow_negative and number < 0:
        return False, None, f"Negative value {number} is not permitted"
    return True, number, ''


ARTICLE5_FIELDS = [
    'a21_art5_lawfulness_fairness',
    'a22_art5_purpose_limitation',
    'a23_art5_data_minimization',
    'a24_art5_accuracy',
    'a25_art5_storage_limitation',
    'a26_art5_integrity_confidentiality',
    'a27_art5_accountability',
]

RIGHT_TO_ARTICLE_MAP = {
    'a37_right_access_violated': {'15'},
    'a38_right_rectification_violated': {'16'},
    'a39_right_erasure_violated': {'17'},
    'a40_right_restriction_violated': {'18'},
    'a41_right_portability_violated': {'20'},
    'a42_right_object_violated': {'21'},
    'a43_transparency_violated': {'12', '13', '14'},
    'a44_automated_decisions_violated': {'22'},
}


def apply_additional_checks(row: dict) -> list[dict]:
    """Run cross-field consistency checks that escalate to hard errors."""

    errors = []

    fine_imposed = row.get('a53_fine_imposed')
    fine_amount = row.get('a54_fine_amount', '')
    if fine_imposed == 'YES':
        ok_amount, parsed_amount, amount_error = parse_int(fine_amount)
        if not ok_amount or parsed_amount is None or parsed_amount <= 0:
            errors.append({
                'field_name': 'a54_fine_amount',
                'field_value': fine_amount,
                'validation_rule': 'consistency',
                'error_message': amount_error or 'a54_fine_amount must be a positive integer when a53_fine_imposed = YES',
                'severity': 'ERROR',
            })

        fine_currency = row.get('a55_fine_currency', '')
        if fine_currency in {'', 'NOT_APPLICABLE', 'NOT_DISCUSSED'}:
            errors.append({
                'field_name': 'a55_fine_currency',
                'field_value': fine_currency,
                'validation_rule': 'consistency',
                'error_message': 'a55_fine_currency must specify an ISO code when a53_fine_imposed = YES',
                'severity': 'ERROR',
            })

    breached_articles = parse_article_numbers(row.get('a77_articles_breached', ''))

    if any(row.get(field) == 'BREACHED' for field in ARTICLE5_FIELDS):
        if '5' not in breached_articles:
            errors.append({
                'field_name': 'a77_articles_breached',
                'field_value': row.get('a77_articles_breached', ''),
                'validation_rule': 'consistency',
                'error_message': 'Article 5 breach flagged (a21-a27) but Art. 5 missing from a77_articles_breached',
                'severity': 'ERROR',
            })

    for field, required_articles in RIGHT_TO_ARTICLE_MAP.items():
        if row.get(field) == 'YES' and breached_articles.isdisjoint(required_articles):
            errors.append({
                'field_name': 'a77_articles_breached',
                'field_value': row.get('a77_articles_breached', ''),
                'validation_rule': 'consistency',
                'error_message': (
                    f"{field} = YES requires one of Articles {', '.join(sorted(required_articles))} in a77_articles_breached"
                ),
                'severity': 'ERROR',
            })

    return errors

# Conditional validation rules
CONDITIONAL_RULES = [
    {
        'field': 'a10_gov_level',
        'condition_field': 'a8_defendant_class',
        'condition_value_not': 'PUBLIC',
        'expected': 'NOT_APPLICABLE',
        'message': 'a10_gov_level should be NOT_APPLICABLE when defendant_class != PUBLIC'
    },
    {
        'field': 'a13_sector_other',
        'condition_field': 'a12_sector',
        'condition_value_not': 'OTHER',
        'expected': 'NOT_APPLICABLE',
        'message': 'a13_sector_other must be NOT_APPLICABLE when sector != OTHER'
    },
    {
        'field': 'a19_art33_breached',
        'condition_field': 'a18_art33_discussed',
        'condition_value': 'NO',
        'expected': 'NOT_APPLICABLE',
        'message': 'a19_art33_breached should be NOT_APPLICABLE when art33_discussed = NO'
    },
    {
        'field': 'a54_fine_amount',
        'condition_field': 'a53_fine_imposed',
        'condition_value': 'NO',
        'expected': 'NOT_APPLICABLE',
        'message': 'a54_fine_amount must be NOT_APPLICABLE when fine_imposed = NO',
        'severity': 'ERROR'
    },
    {
        'field': 'a55_fine_currency',
        'condition_field': 'a53_fine_imposed',
        'condition_value': 'NO',
        'expected': 'NOT_APPLICABLE',
        'message': 'a55_fine_currency must be NOT_APPLICABLE when fine_imposed = NO',
        'severity': 'ERROR'
    },
    {
        'field': 'a57_turnover_amount',
        'condition_field': 'a56_turnover_discussed',
        'condition_value': 'NO',
        'expected': 'NOT_DISCUSSED',
        'message': 'a57_turnover_amount must be NOT_DISCUSSED when turnover_discussed = NO'
    },
    {
        'field': 'a58_turnover_currency',
        'condition_field': 'a56_turnover_discussed',
        'condition_value': 'NO',
        'expected': 'NOT_DISCUSSED',
        'message': 'a58_turnover_currency must be NOT_DISCUSSED when turnover_discussed = NO'
    },
    {
        'field': 'a73_oss_role',
        'condition_field': 'a72_cross_border_oss',
        'condition_value_not': 'YES',
        'expected': 'NOT_APPLICABLE',
        'message': 'a73_oss_role must be NOT_APPLICABLE when cross_border_oss != YES'
    }
]


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

class FieldValidator:
    """Validates individual field values against schema rules."""

    @staticmethod
    def validate_enum(value: str, allowed: list) -> tuple[bool, str, str]:
        """Validate enum field."""
        if value in allowed:
            return (True, '', 'VALID')
        return (False, f"Invalid enum value '{value}'. Allowed: {', '.join(allowed)}", 'ERROR')

    @staticmethod
    def validate_integer_range(
        value: str,
        min_val: int,
        max_val: int,
        warn_outside: bool = False,
        future_tolerance_years: Optional[int] = None,
    ) -> tuple[bool, str, str]:
        """Validate integer within range with optional future tolerance."""
        ok, number, parse_error = parse_int(value, allow_negative=False)
        if not ok or number is None:
            return (False, parse_error, 'ERROR')

        current_year = datetime.utcnow().year
        if future_tolerance_years is not None:
            future_limit = current_year + future_tolerance_years
            if number > future_limit:
                return (
                    False,
                    f"Integer {number} exceeds allowed future tolerance (max {future_limit})",
                    'ERROR',
                )
            if number > current_year:
                return (
                    False,
                    f"Integer {number} is in the future relative to data vintage ({current_year})",
                    'WARNING',
                )

        if min_val <= number <= max_val:
            return (True, '', 'VALID')

        severity = 'WARNING' if warn_outside else 'ERROR'
        return (False, f"Integer {number} outside valid range [{min_val}, {max_val}]", severity)

    @staticmethod
    def validate_free_text(
        value: str,
        non_empty: bool = False,
        *,
        max_length: Optional[int] = None,
        forbidden_patterns: Optional[list[re.Pattern]] = None,
    ) -> tuple[bool, str, str]:
        """Validate free text field with length and placeholder checks."""
        text = value or ''
        if non_empty and not text.strip():
            return (False, "Empty text field (required non-empty)", 'ERROR')

        disallowed = [ch for ch in text if ord(ch) < 32 and ch not in {'\t', '\n', '\r'}]
        if disallowed:
            return (False, 'Control characters detected in text field', 'ERROR')

        stripped = text.strip()
        if max_length is not None and len(stripped) > max_length:
            return (
                False,
                f"Text exceeds configured limit of {max_length} characters",
                'WARNING',
            )

        patterns = forbidden_patterns if forbidden_patterns is not None else DEFAULT_FREE_TEXT_PATTERNS
        for pattern in patterns:
            if pattern.search(text):
                return (
                    False,
                    f"Suspicious placeholder text matches pattern '{pattern.pattern}'",
                    'WARNING',
                )

        return (True, '', 'VALID')

    @staticmethod
    def validate_free_text_or_sentinel(
        value: str,
        sentinel: str,
        *,
        max_length: Optional[int] = None,
        forbidden_patterns: Optional[list[re.Pattern]] = None,
    ) -> tuple[bool, str, str]:
        """Validate free text or sentinel value."""
        if value == sentinel:
            return (True, '', 'VALID')

        return FieldValidator.validate_free_text(
            value,
            False,
            max_length=max_length,
            forbidden_patterns=forbidden_patterns,
        )

    @staticmethod
    def validate_integer_or_sentinel(value: str, sentinel: str) -> tuple[bool, str, str]:
        """Validate integer or sentinel."""
        if value == sentinel:
            return (True, '', 'VALID')
        ok, _, parse_error = parse_int(value)
        if ok:
            return (True, '', 'VALID')
        return (False, parse_error or f"Not integer or '{sentinel}': '{value}'", 'ERROR')

    @staticmethod
    def validate_integer_or_sentinels(value: str, sentinels: list) -> tuple[bool, str, str]:
        """Validate integer or one of multiple sentinels."""
        if value in sentinels:
            return (True, '', 'VALID')
        ok, _, parse_error = parse_int(value)
        if ok:
            return (True, '', 'VALID')
        return (
            False,
            parse_error or f"Not integer or sentinel ({', '.join(sentinels)}): '{value}'",
            'ERROR',
        )

    @staticmethod
    def validate_semicolon_list_or_sentinel(value: str, allowed_tokens: list, sentinel: str) -> tuple[bool, str, str]:
        """Validate semicolon-separated list or sentinel."""
        if value == sentinel:
            return (True, '', 'VALID')

        tokens = [t.strip() for t in value.split(';')]
        invalid_tokens = [t for t in tokens if t and t not in allowed_tokens]

        if invalid_tokens:
            return (False, f"Invalid tokens: {', '.join(invalid_tokens)}", 'ERROR')
        return (True, '', 'VALID')

    @staticmethod
    def validate_semicolon_integers_or_sentinels(value: str, sentinels: list) -> tuple[bool, str, str]:
        """Validate semicolon-separated integers or sentinels."""
        if value in sentinels:
            return (True, '', 'VALID')

        tokens = [t.strip() for t in value.split(';')]
        invalid_tokens = []
        for token in tokens:
            if not token:
                continue
            normalized = normalize_article_token(token)
            ok, _, _ = parse_int(normalized)
            if ok:
                continue

            invalid_tokens.append(token)

        if invalid_tokens:
            return (False, f"Invalid article numbers: {', '.join(invalid_tokens)}", 'ERROR')
        return (True, '', 'VALID')


def validate_row(row: dict) -> list[dict]:
    """
    Validate all fields in a row.
    Returns list of error dictionaries.
    """
    errors = []

    # Validate each field
    for field_name, rules in VALIDATION_RULES.items():
        if field_name not in row:
            errors.append({
                'field_name': field_name,
                'field_value': '',
                'validation_rule': 'field_existence',
                'error_message': 'Field missing from row',
                'severity': 'ERROR'
            })
            continue

        value = row[field_name]
        validator = FieldValidator()
        valid = True
        error_msg = ''
        severity = 'ERROR'

        # Apply validation based on type
        val_type = rules['type']

        if val_type == 'enum':
            valid, error_msg, severity = validator.validate_enum(value, rules['allowed'])

        elif val_type == 'integer_range':
            valid, error_msg, severity = validator.validate_integer_range(
                value,
                rules['min'],
                rules['max'],
                rules.get('warn_outside', False),
                rules.get('future_tolerance_years'),
            )

        elif val_type == 'free_text':
            valid, error_msg, severity = validator.validate_free_text(
                value,
                rules.get('non_empty', False),
                max_length=rules.get('max_length'),
                forbidden_patterns=rules.get('forbidden_patterns'),
            )

        elif val_type == 'free_text_or_sentinel':
            valid, error_msg, severity = validator.validate_free_text_or_sentinel(
                value,
                rules['sentinel'],
                max_length=rules.get('max_length'),
                forbidden_patterns=rules.get('forbidden_patterns'),
            )

        elif val_type == 'integer_or_sentinel':
            valid, error_msg, severity = validator.validate_integer_or_sentinel(value, rules['sentinel'])

        elif val_type == 'integer_or_sentinels':
            valid, error_msg, severity = validator.validate_integer_or_sentinels(value, rules['sentinels'])

        elif val_type == 'semicolon_list_or_sentinel':
            valid, error_msg, severity = validator.validate_semicolon_list_or_sentinel(
                value, rules['allowed_tokens'], rules['sentinel']
            )

        elif val_type == 'semicolon_integers_or_sentinels':
            valid, error_msg, severity = validator.validate_semicolon_integers_or_sentinels(
                value, rules['sentinels']
            )

        if not valid:
            if field_name in BINARY_ONLY_FIELDS and value in FORBIDDEN_BINARY_SENTINELS:
                error_msg = f"Binary field expects YES or NO; found '{value}'"
            errors.append({
                'field_name': field_name,
                'field_value': value,
                'validation_rule': val_type,
                'error_message': error_msg,
                'severity': severity
            })

    # Check conditional rules
    for cond_rule in CONDITIONAL_RULES:
        field = cond_rule['field']
        cond_field = cond_rule['condition_field']
        expected = cond_rule['expected']

        if field not in row or cond_field not in row:
            continue

        # Check condition
        condition_met = False
        if 'condition_value' in cond_rule:
            condition_met = row[cond_field] == cond_rule['condition_value']
        elif 'condition_value_not' in cond_rule:
            condition_met = row[cond_field] != cond_rule['condition_value_not']

        # If condition met, check if field has expected value
        if condition_met and row[field] != expected:
            errors.append({
                'field_name': field,
                'field_value': row[field],
                'validation_rule': 'conditional',
                'error_message': cond_rule['message'],
                'severity': cond_rule.get('severity', 'WARNING')
            })

    errors.extend(apply_additional_checks(row))

    return errors


def validate_dataset(input_file: str):
    """Validate entire dataset and return results."""

    # Read input
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Validating {len(rows)} rows across all 77 fields...")

    all_errors = []
    valid_rows = []
    error_rows_ids = set()

    for row in rows:
        row_id = row.get('id', 'unknown')
        row_errors = validate_row(row)

        if row_errors:
            # Add row ID to each error
            for error in row_errors:
                error['row_id'] = row_id
                all_errors.append(error)
            error_rows_ids.add(row_id)
        else:
            valid_rows.append(row)

    print("✓ Validation complete")
    print(f"  Valid rows: {len(valid_rows)}")
    print(f"  Rows with errors: {len(error_rows_ids)}")
    print(f"  Total errors: {len(all_errors)}")

    return valid_rows, all_errors, rows


def write_outputs(
    valid_rows,
    all_errors,
    all_rows,
    output_validated,
    output_errors,
    output_report,
    schema_hash: str,
    schema_files: list[Path],
):
    """Write validation outputs."""

    # Write validated data
    if valid_rows:
        with open(output_validated, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=valid_rows[0].keys())
            writer.writeheader()
            writer.writerows(valid_rows)
        print(f"✓ Wrote {len(valid_rows)} valid rows to {Path(output_validated).name}")
    else:
        print("⚠ No valid rows to write")

    # Write validation errors
    if all_errors:
        error_fieldnames = [
            'row_id',
            'field_name',
            'field_value',
            'validation_rule',
            'error_message',
            'severity',
            'schema_hash',
        ]
        with open(output_errors, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=error_fieldnames)
            writer.writeheader()
            for error in all_errors:
                writer.writerow({**error, 'schema_hash': schema_hash})
        print(f"✓ Wrote {len(all_errors)} validation errors to {Path(output_errors).name}")

    base_dir = Path(output_validated).parent
    snapshot_path = write_schema_snapshot(base_dir, schema_hash, schema_files)
    print(f"✓ Recorded schema snapshot to {snapshot_path.name}")

    # Generate validation report
    generate_report(
        valid_rows,
        all_errors,
        all_rows,
        output_report,
        output_validated,
        output_errors,
        schema_hash,
        schema_files,
        snapshot_path,
    )


def generate_report(
    valid_rows,
    all_errors,
    all_rows,
    output_report,
    output_validated,
    output_errors,
    schema_hash: str,
    schema_files: list[Path],
    snapshot_path: Path,
):
    """Generate detailed validation report."""

    # Count errors by field
    errors_by_field = defaultdict(int)
    errors_by_severity = defaultdict(int)
    for error in all_errors:
        errors_by_field[error['field_name']] += 1
        errors_by_severity[error['severity']] += 1

    # Top errors
    top_errors = sorted(errors_by_field.items(), key=lambda x: x[1], reverse=True)[:20]

    with open(output_report, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Phase 2: Dataset Validation Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Dataset: {len(all_rows)} rows, 77 fields each\n\n")
        f.write(f"Schema hash: {schema_hash}\n")
        f.write(f"Schema snapshot: {snapshot_path}\n")
        f.write("Schema files:\n")
        for schema_path in schema_files:
            f.write(f"  - {schema_path}\n")
        f.write("\n")

        f.write("VALIDATION SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Valid rows (passed all checks):     {len(valid_rows)}\n")
        f.write(f"Rows with errors:                   {len(all_rows) - len(valid_rows)}\n")
        f.write(f"Total validation errors:            {len(all_errors)}\n")
        f.write(f"  - ERRORS (schema violations):     {errors_by_severity.get('ERROR', 0)}\n")
        f.write(f"  - WARNINGS (suspicious values):   {errors_by_severity.get('WARNING', 0)}\n\n")

        f.write("TOP 20 FIELDS WITH ERRORS\n")
        f.write("-" * 70 + "\n")
        for field, count in top_errors:
            f.write(f"{field:40s} {count:4d} errors\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Detailed errors written to: {Path(output_errors)}\n")
        f.write(f"Clean validated data written to: {Path(output_validated)}\n")
        f.write("=" * 70 + "\n")

    print(f"✓ Wrote validation report to {Path(output_report).name}")


def run_phase2(
    input_file: Path,
    output_validated: Optional[Path] = None,
    output_errors: Optional[Path] = None,
    output_report: Optional[Path] = None,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute phase 2 validation with configurable input/output paths."""

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_path = Path(input_file)

    if output_validated is None or output_errors is None or output_report is None:
        default_input = project_root / 'outputs' / 'phase1_extraction' / 'main_dataset.csv'
        if input_path.resolve() == default_input.resolve():
            base_dir = project_root / 'outputs' / 'phase2_validation'
            output_validated = base_dir / 'validated_data.csv'
            output_errors = base_dir / 'validation_errors.csv'
            output_report = base_dir / 'validation_report.txt'
        else:
            base_dir = input_path.parent
            stem = input_path.stem
            output_validated = base_dir / f"{stem}_validated.csv"
            output_errors = base_dir / f"{stem}_validation_errors.csv"
            output_report = base_dir / f"{stem}_validation_report.txt"

    if output_validated is None or output_errors is None or output_report is None:
        raise ValueError("Output paths could not be determined for validation results.")

    output_validated.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("Phase 2: Dataset Validation (All 77 Fields)")
        print("=" * 70)
        print(f"Input:  {input_path}")
        print(f"Output: {output_validated.parent}")
        print()

    valid_rows, all_errors, all_rows = validate_dataset(str(input_path))

    if verbose:
        print()

    schema_files = resolve_schema_files(project_root)
    schema_hash = compute_schema_hash(project_root)

    write_outputs(
        valid_rows,
        all_errors,
        all_rows,
        str(output_validated),
        str(output_errors),
        str(output_report),
        schema_hash,
        schema_files,
    )

    if verbose:
        print()
        print("=" * 70)
        print("✓ Phase 2 validation complete!")
        print("=" * 70)

    return {
        'input_file': input_path,
        'validated_csv': output_validated if valid_rows else None,
        'errors_csv': output_errors if all_errors else None,
        'report_file': output_report,
        'valid_count': len(valid_rows),
        'error_count': len(all_errors),
        'row_count': len(all_rows),
        'schema_hash': schema_hash,
    }


def main():
    import sys

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if len(sys.argv) > 1 and sys.argv[1] == '--input':
        input_file = Path(sys.argv[2]) if sys.argv[2].startswith('/') else project_root / sys.argv[2]
    else:
        input_file = project_root / 'outputs' / 'phase1_extraction' / 'main_dataset.csv'

    run_phase2(Path(input_file), verbose=True)


if __name__ == '__main__':
    main()
