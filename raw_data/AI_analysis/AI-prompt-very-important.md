# GDPR Decision Data Extraction - Question/Answer Format

## Instructions

Extract facts from one GDPR/data protection authority decision. Output EXACTLY **73 lines** in English. Each line starts with "Answer N: " (where N is 1-73) and contains ONLY the allowed value. No extra text, no explanations, no commentary.

**Critical rules:**
- Answer in ENGLISH regardless of decision language
- Use ONLY the allowed values shown for each question
- Values are CASE-SENSITIVE (e.g., BREACHED not breached)
- Base answers on explicit statements in the decision
- Do NOT infer violations unless the schema explicitly permits interpretation
- If something is not mentioned, use NOT_DISCUSSED (or NOT_APPLICABLE where structurally irrelevant)
- Do NOT quote sentinel codes (NOT_APPLICABLE, NOT_DISCUSSED) - output them plain
- Quote text fields only if they contain commas, quotes, or newlines (use double quotes and escape internal quotes by doubling)

---

## SECTION 1: BASIC CASE METADATA

**Question 1:** What is the country of the deciding authority?  
**Answer 1:** Two-letter ISO code  
Allowed values: AT or BE or BG or HR or CY or CZ or DE or DK or EE or ES or FI or FR or GB or GR or HU or IE or IS or IT or LI or LT or LU or LV or MT or NL or NO or PL or PT or RO or SE or SI or SK

**Question 2:** What is the name of the deciding authority?  
**Answer 2:** Official name as stated in decision (e.g., "CNIL" or "Norwegian Data Protection Authority")

**Question 3:** Is this an appellate decision?  
- YES if court appellate judgment or appeal-tier decision
- NO if first-instance DPA decision or first-instance court ruling
- NOT_DISCUSSED only if posture is unclear  
**Answer 3:** YES or NO or NOT_DISCUSSED

**Question 4:** What is the decision year?  
**Answer 4:** Four-digit year (e.g., 2023)

**Question 5:** What is the decision month?  
- Use 0 if only year given or month unknown
- Use 1-12 if month specified  
**Answer 5:** Integer from 0 to 12

---

## SECTION 2: DEFENDANT INFORMATION

**Question 6:** How many defendants are explicitly named?  
- If multiple defendants, code all subsequent fields for the primary/main defendant only  
**Answer 6:** Integer ≥1

**Question 7:** What is the name of the main defendant?  
**Answer 7:** Official name as stated

**Question 8:** What class is the defendant?  
**Answer 8:** PUBLIC or PRIVATE or NGO_OR_NONPROFIT or RELIGIOUS or INDIVIDUAL or OTHER

**Question 9:** What is the enterprise size?  
- Use UNKNOWN if it cannot be confidently determined; SME means it employs fewer than 250 people and has either an annual turnover of no more than €50 million or a balance sheet total not exceeding €43 million.
- Use NOT_APPLICABLE for non-enterprises (e.g., government ministries, individuals)  
**Answer 9:** SME or LARGE or VERY_LARGE or UNKNOWN or NOT_APPLICABLE

**Question 10:** If PUBLIC defendant, what governmental level?  
- Use NOT_APPLICABLE if defendant_class ≠ PUBLIC  
**Answer 10:** STATE or MUNICIPAL or JUDICIAL or UNKNOWN or NOT_APPLICABLE

**Question 11:** What is the defendant's primary role?  
- Code what the decision explicitly assigns
- JOINT_CONTROLLER only if joint controllers with another entity
- If text mixes roles, choose the role tied to the infringement being decided; if unclear, use NOT_MENTIONED  
**Answer 11:** CONTROLLER or PROCESSOR or JOINT_CONTROLLER or NOT_MENTIONED

**Question 12:** What is the primary sector?  
**Answer 12:** HEALTH or EDUCATION or TELECOM or RETAIL or DIGITAL_SERVICES or FINANCIAL or CONSULTING or HOUSING_TOURISM or FITNESS_WELLNESS or MEDIA or MANUFACTURING or UTILITIES or MEMBERSHIP_ORGS or RELIGIOUS_ORGS or TAX or OTHER_PUBLIC_ADMIN or OTHER

**Question 13:** If sector=OTHER, provide short keywords (semicolon-separated; no spaces after semicolons).  
- Use NOT_APPLICABLE if sector ≠ OTHER  
**Answer 13:** keywords or NOT_APPLICABLE

---

## SECTION 3: PROCESSING CONTEXT

**Question 14:** Do any of these specific processing contexts processing contexts apply to this case?  
- List all that apply, semicolon-separated (e.g., CCTV;EMPLOYEE_MONITORING)
- Use NOT_DISCUSSED if no specific processing context is identifiable
- Do not overly infer - stick to what's clearly present in the decision
**Answer 14:** CCTV or MARKETING or RECRUITMENT_AND_HR or COOKIES or CUSTOMER_LOYALTY_CLUBS or CREDIT_SCORING or BACKGROUND_CHECKS or ARTIFICIAL_INTELLIGENCE or PROBLEMATIC_THIRD_PARTY_SHARING_STATED or DPO_ROLE_PROBLEMS_STATED or JOURNALISM

---

## SECTION 4: CASE ORIGINS

**Question 15:** Did the case involve a direct data subject complaint?  
**Answer 15:** YES or NO or NOT_DISCUSSED

**Question 16:** Was it stated that there was media attention to the case?  
- YES if media attention is mentioned 
- NO if decision states there was no media attention
- NOT_DISCUSSED if decision is silent on this  
**Answer 16:** YES or NO or NOT_DISCUSSED

**Question 17:** Did the DPA expressly conduct an official audit or inspection?  
- Only mark YES if DPA expressly stated it conducted an audit/inspection, typically physically
- Own-initiative investigations without formal audit = NO  
**Answer 17:** YES or NO or NOT_DISCUSSED

---

## SECTION 5: ARTICLE 33 BREACH NOTIFICATION

**Question 18:** Was Article 33 breach notification expressly discussed in the decision?  
**Answer 18:** YES or NO

**Question 19:** Was Article 33 expressly held to be breached?  
- YES only if explicitly found breached
- Use NOT_APPLICABLE if Art 33 not relevant to case (e.g., no breach notification obligation)  
**Answer 19:** YES or NO or NOT_DISCUSSED or NOT_APPLICABLE

**Question 20:** How did breach notification (or lack thereof) affect sanction determination?  
**Answer 20:** AGGRAVATING or MITIGATING or NEUTRAL or NOT_DISCUSSED

---

## SECTION 6: ARTICLE 5 PRINCIPLES

**Guidance for Questions 21-27:** Mark BREACHED only if DPA expressly holds this principle violated. Mark NOT_BREACHED only if DPA explicitly finds compliance. Never infer "not breached" from silence - use NOT_DISCUSSED.

**Question 21:** Article 5(1)(a) Lawfulness/Fairness/Transparency status?  
**Answer 21:** BREACHED or NOT_BREACHED or NOT_DISCUSSED

**Question 22:** Article 5(1)(b) Purpose limitation status?  
**Answer 22:** BREACHED or NOT_BREACHED or NOT_DISCUSSED

**Question 23:** Article 5(1)(c) Data minimization status?  
**Answer 23:** BREACHED or NOT_BREACHED or NOT_DISCUSSED

**Question 24:** Article 5(1)(d) Accuracy status?  
**Answer 24:** BREACHED or NOT_BREACHED or NOT_DISCUSSED

**Question 25:** Article 5(1)(e) Storage limitation status?  
**Answer 25:** BREACHED or NOT_BREACHED or NOT_DISCUSSED

**Question 26:** Article 5(1)(f) Integrity and confidentiality status?  
**Answer 26:** BREACHED or NOT_BREACHED or NOT_DISCUSSED

**Question 27:** Article 5(2) Accountability status?  
**Answer 27:** BREACHED or NOT_BREACHED or NOT_DISCUSSED

---

## SECTION 7: SPECIAL CATEGORIES & VULNERABLE GROUPS

**Question 28:** Was Article 9 special categories of data expressly discussed?  
**Answer 28:** YES or NO

**Question 29:** Which vulnerable groups are expressly involved as data subjects?  
- List all that apply, semicolon-separated
- Mark PRESENT only if expressly mentioned
- CHILDREN = minors/persons under 18
- OTHER_VULNERABLE = disabled, migrants, refugees, etc.
- Use NOT_DISCUSSED if no vulnerable groups mentioned  
**Answer 29:** CHILDREN or EMPLOYEES or ELDERLY or PATIENTS or STUDENTS or OTHER_VULNERABLE or NOT_DISCUSSED

---

## SECTION 8: ARTICLE 6 LEGAL BASES

**Guidance for Questions 30-35:** Code what's expressly assessed by the DPA. Mark VALID only if DPA explicitly finds it valid. Mark INVALID only if DPA explicitly invalidates it. Use NOT_DISCUSSED if DPA doesn't assess this legal basis.

**Question 30:** Article 6(1)(a) Consent status?  
**Answer 30:** VALID or INVALID or NOT_DISCUSSED

**Question 31:** Article 6(1)(b) Contract status?  
**Answer 31:** VALID or INVALID or NOT_DISCUSSED

**Question 32:** Article 6(1)(c) Legal obligation status?  
**Answer 32:** VALID or INVALID or NOT_DISCUSSED

**Question 33:** Article 6(1)(d) Vital interests status?  
**Answer 33:** VALID or INVALID or NOT_DISCUSSED

**Question 34:** Article 6(1)(e) Public task status?  
**Answer 34:** VALID or INVALID or NOT_DISCUSSED

**Question 35:** Article 6(1)(f) Legitimate interest status?  
**Answer 35:** VALID or INVALID or NOT_DISCUSSED

**Question 36:** Provide a brief summary (maximum 3 sentences) of the DPA's legal basis discussion, if any.  
- Outline what legal basis the controller claimed, what the DPA found, and key reasoning
- If legal bases not discussed, state: NOT_DISCUSSED
- Quote if contains commas  
**Answer 36:** Text or NOT_DISCUSSED

---

## SECTION 9: DATA SUBJECT RIGHTS VIOLATIONS

**Guidance for Questions 37-44:** Mark YES only if DPA expressly finds violation. Mark NO only if DPA expressly finds no violation. Use NOT_DISCUSSED if right not assessed.

**Question 37:** Right of access (Article 15) violated?  
**Answer 37:** YES or NO or NOT_DISCUSSED

**Question 38:** Right to rectification (Article 16) violated?  
**Answer 38:** YES or NO or NOT_DISCUSSED

**Question 39:** Right to erasure (Article 17) violated?  
**Answer 39:** YES or NO or NOT_DISCUSSED

**Question 40:** Right to restriction (Article 18) violated?  
**Answer 40:** YES or NO or NOT_DISCUSSED

**Question 41:** Right to data portability (Article 20) violated?  
**Answer 41:** YES or NO or NOT_DISCUSSED

**Question 42:** Right to object (Article 21) violated?  
**Answer 42:** YES or NO or NOT_DISCUSSED

**Question 43:** Transparency/Information obligations (Articles 12-14) violated?  
**Answer 43:** YES or NO or NOT_DISCUSSED

**Question 44:** Rights re automated decisions (Article 22) violated?  
**Answer 44:** YES or NO or NOT_DISCUSSED

---

## SECTION 10: ARTICLE 58 CORRECTIVE MEASURES

**Guidance for Questions 45-52:** Mark YES only if the decision expressly states this as a formal corrective measure under Article 58(2). The DPA saying "you should do X" or "this was problematic" is NOT a formal corrective measure - only official orders/warnings/reprimands count.

**Question 45:** Article 58(2)(a) Warning issued?  
- Only YES if expressly stated as formal warning under Art 58(2)(a)  
**Answer 45:** YES or NO or NOT_DISCUSSED

**Question 46:** Article 58(2)(b) Reprimand issued?  
- Only YES if expressly stated as formal reprimand under Art 58(2)(b)  
**Answer 46:** YES or NO or NOT_DISCUSSED

**Question 47:** Article 58(2)(c) Order to comply with data subject request?  
**Answer 47:** YES or NO or NOT_DISCUSSED

**Question 48:** Article 58(2)(d) Order to bring processing into compliance?  
**Answer 48:** YES or NO or NOT_DISCUSSED

**Question 49:** Article 58(2)(e) Order to communicate breach to data subjects?  
**Answer 49:** YES or NO or NOT_DISCUSSED

**Question 50:** Article 58(2) Erasure/restriction order issued?  
**Answer 50:** YES or NO or NOT_DISCUSSED

**Question 51:** Certification withdrawal?  
**Answer 51:** YES or NO or NOT_DISCUSSED

**Question 52:** Article 58(2)(j) Suspension of data flows?  
**Answer 52:** YES or NO or NOT_DISCUSSED

---

## SECTION 11: FINANCIAL PENALTIES (ORIGINAL CURRENCY ONLY)

**Question 53:** Was an administrative fine imposed?  
**Answer 53:** YES or NO

**Question 54:** Fine amount in original currency?  
- Integer only, no thousands separators, no currency symbols
- Use NOT_APPLICABLE if fine_issued=NO  
**Answer 54:** Integer or NOT_APPLICABLE

**Question 55:** Currency of fine (ISO-4217 code)?  
- E.g., EUR or GBP or SEK
- Use NOT_APPLICABLE if fine_issued=NO  
**Answer 55:** Currency code or NOT_APPLICABLE

**Question 56:** Was defendant's turnover expressly discussed by the DPA?  
**Answer 56:** YES or NO

**Question 57:** Turnover amount in original currency stated in decision?  
- Integer only, no separators
- Use NOT_DISCUSSED if turnover_discussed=NO
- Use NOT_APPLICABLE if turnover concept doesn't fit defendant type (e.g., individuals, government ministries)  
**Answer 57:** Integer or NOT_APPLICABLE or NOT_DISCUSSED

**Question 58:** Currency of turnover (ISO-4217 code)?  
- Use NOT_DISCUSSED if turnover_discussed=NO
- Use NOT_APPLICABLE if turnover_eur=NOT_APPLICABLE  
**Answer 58:** Currency code or NOT_APPLICABLE or NOT_DISCUSSED

---

## SECTION 12: ARTICLE 83(2) FACTORS

**Guidance for Questions 59-69:** These measure HOW each factor affected sanction determination:
- AGGRAVATING = DPA discussed this factor as increasing seriousness or fine amount (e.g., "weighs against", "enhances gravity", "prolonged duration", "moderately grave")
- MITIGATING = DPA discussed this factor as reducing seriousness or fine amount (e.g., "weighs in favor", "took remedial actions", "reduces fine")
- NEUTRAL = DPA discussed factor but explicitly stated it doesn't affect outcome (e.g., "weighs neither for nor against", "normal circumstance", "cannot be considered mitigating")
- NOT_DISCUSSED = DPA did not mention this criterion

If DPA discusses a factor negatively/positively without using "aggravating/mitigating," interpret that based on context.

**Question 59:** Article 83(2)(a) Nature/gravity/duration?  
**Answer 59:** AGGRAVATING or MITIGATING or NEUTRAL or NOT_DISCUSSED

**Question 60:** Article 83(2)(b) Intentional or negligent character?  
**Answer 60:** AGGRAVATING or MITIGATING or NEUTRAL or NOT_DISCUSSED

**Question 61:** Article 83(2)(c) Actions to mitigate damage?  
**Answer 61:** AGGRAVATING or MITIGATING or NEUTRAL or NOT_DISCUSSED

**Question 62:** Article 83(2)(d) Technical and organisational measures?  
**Answer 62:** AGGRAVATING or MITIGATING or NEUTRAL or NOT_DISCUSSED

**Question 63:** Article 83(2)(e) Previous infringements?  
**Answer 63:** AGGRAVATING or MITIGATING or NEUTRAL or NOT_DISCUSSED

**Question 64:** Article 83(2)(f) Cooperation with supervisory authority?  
**Answer 64:** AGGRAVATING or MITIGATING or NEUTRAL or NOT_DISCUSSED

**Question 65:** Article 83(2)(g) Categories of personal data affected?  
**Answer 65:** AGGRAVATING or MITIGATING or NEUTRAL or NOT_DISCUSSED

**Question 66:** Article 83(2)(h) Manner in which infringement became known to authority?  
**Answer 66:** AGGRAVATING or MITIGATING or NEUTRAL or NOT_DISCUSSED

**Question 67:** Article 83(2)(i) Compliance with prior orders?  
**Answer 67:** AGGRAVATING or MITIGATING or NEUTRAL or NOT_DISCUSSED

**Question 68:** Article 83(2)(j) Adherence to codes of conduct or certification?  
**Answer 68:** AGGRAVATING or MITIGATING or NEUTRAL or NOT_DISCUSSED

**Question 69:** Article 83(2)(k) Other aggravating or mitigating factors?  
**Answer 69:** AGGRAVATING or MITIGATING or NEUTRAL or NOT_DISCUSSED

**Question 70:** Did the DPA systematically discuss most Article 83(2) factors?  
- YES if DPA explicitly addressed 7 or more of the 11 factors (even if briefly)
- NO if DPA addressed fewer than 7 factors or did not structure discussion around Art 83(2)  
**Answer 70:** YES or NO

**Question 71:** Is this the defendant's first GDPR violation?  
- As stated or implied by the decision  
**Answer 71:** YES_FIRST_TIME or NO_REPEAT_OFFENDER or NOT_DISCUSSED

---

## SECTION 13: CROSS-BORDER & REFERENCES

**Question 72:** Did the case involve One-Stop-Shop cross-border processing?  
**Answer 72:** YES or NO or NOT_DISCUSSED

**Question 73:** What was this authority's OSS role?  
- Use NOT_APPLICABLE if oss_cross_border ≠ YES  
**Answer 73:** LEAD or CONCERNED or NOT_APPLICABLE

**Question 74:** List ALL EDPB and WP29 guidelines/recommendations/opinions referenced in the decision.  
- Semicolon-separated, no quotes around titles
- Include both EDPB and Article 29 Working Party documents
- Use NOT_APPLICABLE if no guidelines referenced
- Quote the entire answer if titles contain commas  
**Answer 74:** Titles or NOT_APPLICABLE

---

## SECTION 14: SUMMARIES

**Question 75:** Case summary (maximum 4 sentences).  
- Short summary of key facts, legal issue, and DPA reasoning
- Written for lawyers already familiar with GDPR - skip basic background
- Zero fluff, expert-skimmable
- Quote if contains commas  
**Answer 75:** Text

**Question 76:** Article 83(2) weighing summary (maximum 4 sentences).  
- State whether Art 83(2) was discussed
- Briefly list what was aggravating, mitigating, and neutral
- If Art 83(2) not discussed, state: NOT_DISCUSSED
- Quote if contains commas  
**Answer 76:** Text or NOT_DISCUSSED

**Question 77:** List GDPR articles found breached (semicolon-separated integers).  
- E.g., 5;6;15;17 or 32;33
- Use NONE_VIOLATED only if decision explicitly states no breach occurred
- Use NOT_DISCUSSED if articles not specified by number
- Quote if needed  
**Answer 77:** Numbers or NONE_VIOLATED or NOT_DISCUSSED

---

# OUTPUT FORMAT

Return exactly 77 lines, each starting with "Answer N: " followed by only the allowed value:

```
Answer 1: DE
Answer 2: Bavarian DPA
Answer 3: NO
Answer 4: 2023
Answer 5: 3
...
Answer 77: 5;6;32
```

No extra text. No explanations. No commentary. No line breaks within individual answers.

═══════════════════════════════════════════════════

## DECISION TEXT TO ANALYZE

═══════════════════════════════════════════════════