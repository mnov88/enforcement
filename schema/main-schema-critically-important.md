### Section 1: Basic Case Metadata

| Field | Allowed Values | Validation Type |
|-------|---------------|-----------------|
| `a1_country_code` | AT, BE, BG, HR, CY, CZ, DE, DK, EE, EL, ES, FI, FR, GB, GR, HU, IE, IS, IT, LI, LT, LU, LV, MT, NL, NO, PL, PT, RO, SE, SI, SK | Exact enum match (EL = Greece alternative code) |
| `a2_authority_name` | Free text (e.g., "Swedish Privacy Protection Agency") | Non-empty text |
| `a3_appellate_decision` | YES, NO, NOT_DISCUSSED | Exact enum match |
| `a4_decision_year` | 2018-2025 | Numeric range |
| `a5_decision_month` | 0-12 (0 if unknown) | Numeric range |

### Section 2: Defendant Information

| Field | Allowed Values | Notes |
|-------|---------------|-------|
| `a6_num_defendants` | Positive integer ≥ 1 | Must be numeric |
| `a7_defendant_name` | Free text | Required field |
| `a8_defendant_class` | PUBLIC, PRIVATE, NGO_OR_NONPROFIT, RELIGIOUS, INDIVIDUAL, POLITICAL_PARTY, OTHER | Exact match |
| `a9_enterprise_size` | SME, LARGE, VERY_LARGE, UNKNOWN, **NOT_APPLICABLE** | Includes NOT_APPLICABLE |
| `a10_gov_level` | STATE, MUNICIPAL, JUDICIAL, UNKNOWN, **NOT_APPLICABLE** | NOT_APPLICABLE if not public |
| `a11_defendant_role` | CONTROLLER, PROCESSOR, JOINT_CONTROLLER, NOT_MENTIONED | Data protection roles |
| `a12_sector` | HEALTH, EDUCATION, TELECOM, RETAIL, DIGITAL_SERVICES, FINANCIAL, CONSULTING, HOUSING_TOURISM, FITNESS_WELLNESS, MEDIA, MANUFACTURING, UTILITIES, MEMBERSHIP_ORGS, RELIGIOUS_ORGS, TAX, OTHER_PUBLIC_ADMIN, OTHER | Complete sector list |
| `a13_sector_other` | Keywords (semicolon-separated) or **NOT_APPLICABLE** | Only if sector = OTHER |

### Section 3-4: Processing Context and Case Origins

| Field | Allowed Values | Format |
|-------|---------------|--------|
| `a14_processing_contexts` | CCTV, MARKETING, RECRUITMENT_AND_HR, COOKIES, CUSTOMER_LOYALTY_CLUBS, CREDIT_SCORING, BACKGROUND_CHECKS, ARTIFICIAL_INTELLIGENCE, PROBLEMATIC_THIRD_PARTY_SHARING_STATED, DPO_ROLE_PROBLEMS_STATED, JOURNALISM, EMPLOYEE_MONITORING, NOT_DISCUSSED | Semicolon-separated list |
| `a15_data_subject_complaint` | YES, NO, NOT_DISCUSSED | Standard three-option |
| `a16_media_attention` | YES, NO, NOT_DISCUSSED | Standard three-option |
| `a17_official_audit` | YES, NO, NOT_DISCUSSED | Standard three-option |

### Section 5: Article 33 Breach Notification

| Field | Allowed Values | Special Notes |
|-------|---------------|---------------|
| `a18_art33_discussed` | YES, NO | Only two options (no NOT_DISCUSSED) |
| `a19_art33_breached` | YES, NO, NOT_DISCUSSED, **NOT_APPLICABLE** | Four-option field |
| `a20_breach_notification_effect` | AGGRAVATING, MITIGATING, NEUTRAL, NOT_DISCUSSED | Article 83(2) factor |

### Section 6: Article 5 Principles (Questions 21-27)

All Article 5 principle fields use the same validation:

**Fields**: `a21_art5_lawfulness_fairness`, `a22_art5_purpose_limitation`, `a23_art5_data_minimization`, `a24_art5_accuracy`, `a25_art5_storage_limitation`, `a26_art5_integrity_confidentiality`, `a27_art5_accountability`

**Allowed Values**: BREACHED, NOT_BREACHED, NOT_DISCUSSED

### Section 7: Special Categories & Vulnerable Groups

| Field | Allowed Values | Format |
|-------|---------------|--------|
| `a28_art9_discussed` | YES, NO | Two-option field |
| `a29_vulnerable_groups` | CHILDREN, EMPLOYEES, ELDERLY, PATIENTS, STUDENTS, OTHER_VULNERABLE, NOT_DISCUSSED | Semicolon-separated list |

### Section 8: Article 6 Legal Bases (Questions 30-35)

All Article 6 legal basis fields use the same validation:

**Fields**: `a30_art6_consent`, `a31_art6_contract`, `a32_art6_legal_obligation`, `a33_art6_vital_interests`, `a34_art6_public_task`, `a35_art6_legitimate_interest`

**Allowed Values**: VALID, INVALID, NOT_DISCUSSED

| Field | Allowed Values | Format |
|-------|---------------|--------|
| `a36_legal_basis_summary` | Free text or NOT_DISCUSSED | Quote if contains commas |

### Section 9: Data Subject Rights Violations (Questions 37-44)

All data subject rights fields use standard three-option validation:

**Fields**: `a37_right_access_violated`, `a38_right_rectification_violated`, `a39_right_erasure_violated`, `a40_right_restriction_violated`, `a41_right_portability_violated`, `a42_right_object_violated`, `a43_transparency_violated`, `a44_automated_decisions_violated`

**Allowed Values**: YES, NO, NOT_DISCUSSED

### Section 10: Article 58 Corrective Measures (Questions 45-52)

All Article 58 corrective measure fields use standard three-option validation:

**Fields**: `a45_warning_issued`, `a46_reprimand_issued`, `a47_comply_data_subject_order`, `a48_compliance_order`, `a49_breach_communication_order`, `a50_erasure_restriction_order`, `a51_certification_withdrawal`, `a52_data_flow_suspension`

**Allowed Values**: YES, NO, NOT_DISCUSSED

### Section 11: Financial Penalties

| Field | Allowed Values | Special Rules |
|-------|---------------|---------------|
| `a53_fine_imposed` | YES, NO | Two-option field |
| `a54_fine_amount` | Positive integer or **NOT_APPLICABLE** | No separators or symbols |
| `a55_fine_currency` | EUR, GBP, SEK, DKK, NOK, PLN, CZK, HUF, RON, BGN, HRK, CHF, ISK, USD, **NOT_APPLICABLE** | ISO currency codes |
| `a56_turnover_discussed` | YES, NO | Two-option field |
| `a57_turnover_amount` | Positive integer, **NOT_APPLICABLE**, or **NOT_DISCUSSED** | Three options |
| `a58_turnover_currency` | EUR, GBP, SEK, DKK, NOK, PLN, CZK, HUF, RON, BGN, HRK, CHF, ISK, USD, **NOT_APPLICABLE**, **NOT_DISCUSSED** | Currency with special values |

### Section 12: Article 83(2) Factors (Questions 59-69)

All Article 83(2) factor fields use the same validation:

**Fields**: `a59_nature_gravity_duration`, `a60_intentional_negligent`, `a61_mitigate_damage_actions`, `a62_technical_org_measures`, `a63_previous_infringements`, `a64_cooperation_authority`, `a65_data_categories_affected`, `a66_infringement_became_known`, `a67_prior_orders_compliance`, `a68_codes_certification`, `a69_other_factors`

**Allowed Values**: AGGRAVATING, MITIGATING, NEUTRAL, NOT_DISCUSSED

| Field | Allowed Values | Notes |
|-------|---------------|-------|
| `a70_systematic_art83_discussion` | YES, NO | Two-option field |
| `a71_first_violation` | YES_FIRST_TIME, NO_REPEAT_OFFENDER, NOT_DISCUSSED | Special three-option |

### Section 13: Cross-Border & References

| Field | Allowed Values | Special Rules |
|-------|---------------|---------------|
| `a72_cross_border_oss` | YES, NO, NOT_DISCUSSED | Standard three-option |
| `a73_oss_role` | LEAD, CONCERNED, **NOT_APPLICABLE** | NOT_APPLICABLE if no OSS |
| `a74_guidelines_referenced` | Guidelines titles or **NOT_APPLICABLE** | Semicolon-separated list |

### Section 14: Summaries

| Field | Allowed Values | Format Rules |
|-------|---------------|--------------|
| `a75_case_summary` | Free text (max 4 sentences) | Quote if contains commas |
| `a76_art83_weighing_summary` | Free text or **NOT_DISCUSSED** | Quote if contains commas |
| `a77_articles_breached` | Numbers (semicolon-separated), **NONE_VIOLATED**, or **NOT_DISCUSSED** | E.g., "5;6;15" |
