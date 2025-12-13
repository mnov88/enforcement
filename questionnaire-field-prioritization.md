# Questionnaire Field Prioritization for Future Data Collection

## Analysis Context

This document evaluates each of the 77 schema fields against the proposed research methodology ("Reasoning and Consistency in GDPR Enforcement"). Fields are categorized as **ESSENTIAL**, **USEFUL**, or **DROPPABLE** for future data collection efforts.

---

## Summary: Streamlined Questionnaire

| Category | Count | Reduction |
|----------|-------|-----------|
| Current fields | 77 | — |
| **Essential** | 26 | — |
| **Useful** | 21 | — |
| **Droppable** | 30 | 39% reduction |
| **Minimum viable** | 26 | 66% reduction |
| **Recommended** | 47 | 39% reduction |

---

## ESSENTIAL FIELDS (26 fields) — Cannot Drop

These fields are required for core analyses: dependent variables, Article 83(2) factors, matching variables, and key controls.

### Identifiers & Geography (3 fields)
| Field | Rationale |
|-------|-----------|
| `a1_country_code` | Cross-border matching, variance decomposition |
| `a2_authority_name` | Authority-level systematicity index |
| `a4_decision_year` | Temporal controls, period splits |

### Defendant Characteristics (4 fields)
| Field | Rationale |
|-------|-----------|
| `a8_defendant_class` | PUBLIC/PRIVATE comparison, matching covariate |
| `a9_enterprise_size` | Fine scaling control, matching covariate |
| `a11_defendant_role` | CONTROLLER/PROCESSOR distinction |
| `a12_sector` | Sector-specific effects, matching covariate |

### Financial Penalties (3 fields)
| Field | Rationale |
|-------|-----------|
| `a53_fine_imposed` | Sample selection (fine vs. no-fine) |
| `a54_fine_amount` | **Primary dependent variable** |
| `a55_fine_currency` | FX conversion for EUR normalization |

### Article 83(2) Factors (13 fields) — **CORE OF RESEARCH**
| Field | Factor | Rationale |
|-------|--------|-----------|
| `a59_nature_gravity_duration` | Art. 83(2)(a) | Factor effect estimation |
| `a60_intentional_negligent` | Art. 83(2)(b) | Factor effect estimation |
| `a61_mitigate_damage_actions` | Art. 83(2)(c) | Factor effect estimation |
| `a62_technical_org_measures` | Art. 83(2)(d) | Factor effect estimation |
| `a63_previous_infringements` | Art. 83(2)(e) | Factor effect estimation |
| `a64_cooperation_authority` | Art. 83(2)(f) | Factor effect estimation |
| `a65_data_categories_affected` | Art. 83(2)(g) | Factor effect estimation |
| `a66_infringement_became_known` | Art. 83(2)(h) | Factor effect estimation |
| `a67_prior_orders_compliance` | Art. 83(2)(i) | Factor effect estimation |
| `a68_codes_certification` | Art. 83(2)(j) | Factor effect estimation |
| `a69_other_factors` | Art. 83(2)(k) | Factor effect estimation |
| `a70_systematic_art83_discussion` | — | Systematicity validation |
| `a71_first_violation` | Repeat offender | Key control variable |

### Violation Identification (3 fields)
| Field | Rationale |
|-------|-----------|
| `a77_articles_breached` | **Article cohort key** for matching |
| `a21_art5_lawfulness_fairness` | Core Art. 5 breach indicator |
| `a26_art5_integrity_confidentiality` | Security breach indicator |

---

## USEFUL FIELDS (21 fields) — Improve Model, Keep if Feasible

These fields enhance analysis but are not strictly required for core research questions.

### Temporal Refinement (1 field)
| Field | Value Added |
|-------|-------------|
| `a5_decision_month` | Finer temporal controls, seasonality |

### Case Origins (3 fields)
| Field | Value Added |
|-------|-------------|
| `a15_data_subject_complaint` | Selection into enforcement |
| `a17_official_audit` | Proactive vs. reactive enforcement |
| `a3_appellate_decision` | Appeal effects on reasoning |

### Processing Contexts (1 field)
| Field | Value Added |
|-------|-------------|
| `a14_processing_contexts` | Context-specific penalty effects (CCTV, marketing, etc.) |

### Article 5 Detail (5 fields)
| Field | Value Added |
|-------|-------------|
| `a22_art5_purpose_limitation` | Specific principle breach |
| `a23_art5_data_minimization` | Specific principle breach |
| `a24_art5_accuracy` | Specific principle breach |
| `a25_art5_storage_limitation` | Specific principle breach |
| `a27_art5_accountability` | Specific principle breach |

### Legal Basis (3 fields)
| Field | Value Added |
|-------|-------------|
| `a30_art6_consent` | Consent validity effects |
| `a31_art6_contract` | Contract basis effects |
| `a35_art6_legitimate_interest` | Legitimate interest effects |

### Turnover (3 fields)
| Field | Value Added |
|-------|-------------|
| `a56_turnover_discussed` | Proportionality analysis |
| `a57_turnover_amount` | Fine-to-turnover ratio |
| `a58_turnover_currency` | Turnover FX conversion |

### Cross-Border (2 fields)
| Field | Value Added |
|-------|-------------|
| `a72_cross_border_oss` | OSS mechanism analysis |
| `a73_oss_role` | Lead vs. concerned authority |

### Data Breach (2 fields)
| Field | Value Added |
|-------|-------------|
| `a18_art33_discussed` | Breach notification context |
| `a19_art33_breached` | Art. 33 violation indicator |

### Special Categories (1 field)
| Field | Value Added |
|-------|-------------|
| `a28_art9_discussed` | Sensitive data involvement |

---

## DROPPABLE FIELDS (30 fields) — Can Omit for This Research

These fields add minimal value for the proposed research angle and can be safely dropped.

### Low Variance / Sparse (6 fields)
| Field | Why Droppable |
|-------|---------------|
| `a6_num_defendants` | Usually 1; low variance |
| `a10_gov_level` | Only for PUBLIC defendants; sparse |
| `a32_art6_legal_obligation` | Rarely INVALID |
| `a33_art6_vital_interests` | Almost never used |
| `a34_art6_public_task` | Sparse; mostly NOT_DISCUSSED |
| `a51_certification_withdrawal` | Extremely rare measure |

### Identification Only (2 fields)
| Field | Why Droppable |
|-------|---------------|
| `a7_defendant_name` | Identification, not analytic |
| `a13_sector_other` | Free text; covered by a12 |

### Redundant with Other Fields (4 fields)
| Field | Why Droppable |
|-------|---------------|
| `a20_breach_notification_effect` | Derivable from a19 + a59 |
| `a36_legal_basis_summary` | Free text; covered by a30-a35 |
| `a75_case_summary` | Qualitative; not for regression |
| `a76_art83_weighing_summary` | Qualitative supplement to a59-a69 |

### Not Core to Factor/Harmonization Research (10 fields)
| Field | Why Droppable |
|-------|---------------|
| `a16_media_attention` | Rarely informative; high NOT_DISCUSSED |
| `a29_vulnerable_groups` | Different research angle |
| `a37_right_access_violated` | Rights-focused research |
| `a38_right_rectification_violated` | Rights-focused research |
| `a39_right_erasure_violated` | Rights-focused research |
| `a40_right_restriction_violated` | Rights-focused research |
| `a41_right_portability_violated` | Rights-focused research |
| `a42_right_object_violated` | Rights-focused research |
| `a43_transparency_violated` | Rights-focused research |
| `a44_automated_decisions_violated` | Rights-focused research |

### Corrective Measures (8 fields)
| Field | Why Droppable |
|-------|---------------|
| `a45_warning_issued` | Sanction mix research angle |
| `a46_reprimand_issued` | Sanction mix research angle |
| `a47_comply_data_subject_order` | Sanction mix research angle |
| `a48_compliance_order` | Sanction mix research angle |
| `a49_breach_communication_order` | Sanction mix research angle |
| `a50_erasure_restriction_order` | Sanction mix research angle |
| `a52_data_flow_suspension` | Sanction mix research angle |
| `a74_guidelines_referenced` | Text field; hard to operationalize |

---

## Recommended Streamlined Questionnaire (47 fields)

For future data collection focused on this research, use **Essential + Useful** fields:

```
SECTION 1: CASE IDENTIFICATION (5 fields)
a1  Country code
a2  Authority name
a3  Appellate decision [USEFUL]
a4  Decision year
a5  Decision month [USEFUL]

SECTION 2: DEFENDANT (4 fields)
a8  Defendant class
a9  Enterprise size
a11 Defendant role
a12 Sector

SECTION 3: CASE ORIGIN & CONTEXT (3 fields)
a14 Processing contexts [USEFUL]
a15 Data subject complaint [USEFUL]
a17 Official audit [USEFUL]

SECTION 4: BREACH NOTIFICATION (2 fields)
a18 Art. 33 discussed [USEFUL]
a19 Art. 33 breached [USEFUL]

SECTION 5: ARTICLE 5 PRINCIPLES (7 fields)
a21 Lawfulness/fairness
a22 Purpose limitation [USEFUL]
a23 Data minimization [USEFUL]
a24 Accuracy [USEFUL]
a25 Storage limitation [USEFUL]
a26 Integrity/confidentiality
a27 Accountability [USEFUL]

SECTION 6: SPECIAL CATEGORIES (1 field)
a28 Art. 9 discussed [USEFUL]

SECTION 7: LEGAL BASIS (3 fields)
a30 Consent [USEFUL]
a31 Contract [USEFUL]
a35 Legitimate interest [USEFUL]

SECTION 8: FINANCIAL PENALTIES (6 fields)
a53 Fine imposed
a54 Fine amount
a55 Fine currency
a56 Turnover discussed [USEFUL]
a57 Turnover amount [USEFUL]
a58 Turnover currency [USEFUL]

SECTION 9: ARTICLE 83(2) FACTORS (13 fields) — KEEP ALL
a59 Nature, gravity, duration
a60 Intentional/negligent
a61 Mitigate damage actions
a62 Technical/org measures
a63 Previous infringements
a64 Cooperation with authority
a65 Data categories affected
a66 Infringement became known
a67 Prior orders compliance
a68 Codes/certification adherence
a69 Other factors
a70 Systematic Art. 83 discussion
a71 First violation

SECTION 10: CROSS-BORDER (2 fields)
a72 Cross-border OSS [USEFUL]
a73 OSS role [USEFUL]

SECTION 11: ARTICLES BREACHED (1 field)
a77 Articles breached
```

---

## Minimum Viable Questionnaire (26 fields)

For maximum efficiency with minimal quality loss:

```
IDENTIFIERS:        a1, a2, a4
DEFENDANT:          a8, a9, a11, a12
PENALTIES:          a53, a54, a55
ART. 83(2) FACTORS: a59, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a70, a71
VIOLATIONS:         a21, a26, a77
```

**Trade-offs:**
- Loses processing context effects (no a14)
- Loses legal basis analysis (no a30, a31, a35)
- Loses turnover proportionality (no a56-a58)
- Loses OSS mechanism analysis (no a72-a73)
- Retains all core factor analysis and cross-border matching capability

---

## Fields to ADD (if expanding)

If collecting new data, consider these **additions** not in current schema:

| New Field | Rationale |
|-----------|-----------|
| `fine_reduction_percentage` | Settlement/early resolution discount |
| `appeal_outcome` | Whether fine was upheld/reduced/overturned |
| `decision_language` | For full-text analysis feasibility |
| `edpb_opinion_cited` | Harmonization mechanism tracking |
| `fine_calculation_method` | Turnover-based vs. fixed vs. hybrid |
| `concurrent_proceedings` | Multiple DPAs involved |

---

*Document prepared: 2025-12-13*
