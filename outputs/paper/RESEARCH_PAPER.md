# Reasoning and Consistency in GDPR Enforcement

## A Cross-Border Analysis of Penalty Factors and Fine Disparities

---

**Abstract**

The General Data Protection Regulation (GDPR) aims to create uniform data protection enforcement across the European Union. However, whether Data Protection Authorities (DPAs) apply Article 83(2) penalty factors consistently and whether similar violations receive comparable sanctions remains empirically underexplored. Using a novel dataset of 528 fine-imposed decisions across 18 EU/EEA jurisdictions (2018-2024), we apply mixed-effects regression, nearest-neighbor matching, and comprehensive robustness checks to examine enforcement consistency. We find that aggravating factors significantly predict higher fines (β = 0.22, p < 0.0001; ~25% increase per factor), but mitigating factors do not symmetrically reduce penalties (β = -0.04, p = 0.44). Cross-border disparities are pronounced: matched cases with identical article violations show 13-fold average fine differences across jurisdictions (8.7-fold when excluding mega-fines >€10M; Cohen's d = 1.23). Variance decomposition reveals that 60% of fine variance is attributable to country and authority effects, with "which DPA you get" explaining 25% of variance beyond national policy differences (13% excluding mega-fines). These findings challenge GDPR's harmonization objectives and suggest that factor articulation may function as post-hoc justification rather than determinative reasoning.

**Keywords:** GDPR, enforcement consistency, administrative fines, cross-border regulation, mixed-effects models

---

## 1. Introduction

The General Data Protection Regulation (GDPR) represents the European Union's most ambitious attempt to harmonize data protection enforcement across member states. Effective since May 2018, the regulation grants Data Protection Authorities substantial discretion to impose administrative fines up to EUR 20 million or 4% of global annual turnover. Article 83(2) enumerates eleven factors that authorities must consider when determining penalty magnitude, ranging from the nature and gravity of the infringement to the violator's cooperation with supervisory authorities.

Despite its harmonization ambitions, the GDPR's decentralized enforcement model raises questions about consistency. Each member state operates its own DPA with varying resources, traditions, and priorities. Early evidence suggests significant cross-jurisdictional variation in both the frequency and magnitude of penalties (Custers et al., 2022; Adshead et al., 2021). However, systematic empirical analysis of whether this variation reflects case differences or enforcement heterogeneity has been limited by data availability.

This paper addresses four research questions:

1. Do DPAs' articulated Article 83(2) factors predict fine magnitude after controlling for violation characteristics?
2. Are stated aggravating and mitigating reasons systematically applied, or do they appear post-hoc?
3. Do similar violations receive similar penalties across EU member states?
4. How much enforcement variance is attributable to authority-level heterogeneity versus case characteristics?

We analyze 528 fine-imposed GDPR decisions using mixed-effects regression to account for hierarchical authority structure, nearest-neighbor matching to identify comparable cross-border cases, and variance decomposition to quantify enforcement heterogeneity sources. Our findings reveal substantial inconsistencies that challenge the regulation's harmonization goals.

---

## 2. Institutional Background

### 2.1 GDPR Enforcement Framework

The GDPR establishes a cooperative enforcement system through the European Data Protection Board (EDPB), which coordinates DPA activities through consistency mechanisms. The One-Stop-Shop (OSS) mechanism designates a lead supervisory authority for cross-border processing, intended to reduce forum shopping and ensure consistent outcomes.

Article 83 empowers DPAs to impose administrative fines and specifies eleven factors for penalty determination:

1. Nature, gravity, and duration of infringement (Art. 83(2)(a))
2. Intentional or negligent character (Art. 83(2)(b))
3. Actions to mitigate damage (Art. 83(2)(c))
4. Degree of responsibility given technical measures (Art. 83(2)(d))
5. Previous infringements (Art. 83(2)(e))
6. Cooperation with authority (Art. 83(2)(f))
7. Categories of personal data affected (Art. 83(2)(g))
8. How infringement became known (Art. 83(2)(h))
9. Compliance with prior orders (Art. 83(2)(i))
10. Adherence to codes of conduct (Art. 83(2)(j))
11. Other aggravating or mitigating factors (Art. 83(2)(k))

DPAs must encode each applicable factor as aggravating, mitigating, or neutral. However, the regulation provides no quantitative guidance on factor weighting, leaving substantial discretion to individual authorities.

### 2.2 Enforcement Variation Concerns

Academic and policy commentary has noted significant variation in GDPR enforcement (European Parliament, 2021). Ireland, hosting major technology firms' European headquarters, has faced criticism for slow enforcement and moderate penalties despite handling high-profile cases. Meanwhile, France (CNIL) and Germany have imposed substantial fines for comparable violations. Whether these differences reflect case heterogeneity, legitimate policy variation, or problematic inconsistency remains debated.

---

## 3. Theoretical Framework

### 3.1 Regulatory Decision-Making

Administrative penalty determination involves complex trade-offs between deterrence, proportionality, and legal certainty (Becker, 1968; Stigler, 1970). Economic theory suggests optimal penalties should internalize violation costs, but regulatory practice must balance multiple objectives including signaling, rehabilitation, and procedural fairness.

The GDPR's multi-factor framework reflects proportionality principles common in European administrative law. However, the absence of quantitative guidance creates potential for inconsistent application across authorities with different interpretive traditions.

### 3.2 Hypotheses

**H1 (Factor Predictiveness):** Aggravating factor counts positively predict fine magnitude; mitigating factor counts negatively predict fine magnitude, controlling for violation type.

**H2 (Reasoning Systematicity):** Authorities with higher systematicity indices (more complete, directionally consistent factor articulation) produce fines better predicted by case characteristics.

**H3 (Cross-Border Disparities):** Matched cases with identical article violations show significant fine variation across jurisdictions, exceeding variation explained by defendant characteristics.

**H4 (Authority Heterogeneity):** Authority-level random effects account for substantial variance in fine outcomes beyond country-level effects.

---

## 4. Data and Methods

### 4.1 Data Source

We analyze decisions from the GDPR Enforcement Tracker (CMS Law), a comprehensive database of publicly disclosed enforcement actions. Each decision was AI-annotated with 77 structured fields covering case metadata, defendant characteristics, violation specifics, Article 83(2) factor assessments, and penalty outcomes. Annotation followed a detailed schema validated against the regulation's requirements.

### 4.2 Sample Construction

The analytical sample comprises 528 decisions meeting the following criteria:

- Valid schema structure (1,467 of 1,473 extracted records)
- Fine imposed (a53_fine_imposed = "YES")
- Positive fine amount documented
- Decision date between May 2018 and December 2024

For cross-border matching analysis, we further restrict to 316 cases belonging to article cohorts represented in at least two countries, yielding 238 matched pairs.

### 4.3 Variables

**Dependent Variable:** Natural log of fine amount in 2025 real EUR, computed via:
```
log_fine_2025 = ln(fine_amount_eur_real_2025 + 1)
```

Currency conversion used ECB monthly rates; deflation used HICP EA19 annual indices.

**Article 83(2) Factor Variables:** Each factor encoded as AGGRAVATING (+1), MITIGATING (-1), NEUTRAL (0), or NOT_DISCUSSED (NA). Aggregate measures include counts of each category and a balance score (aggravating minus mitigating).

**Control Variables:**
- Defendant class (private, public, other)
- Enterprise size (SME, large, very large)
- Violation type (article breach indicators)
- Case origin (complaint, audit, other)
- Cross-border indicator (OSS case)
- Processing context flags (CCTV, marketing, etc.)

### 4.4 Analytical Methods

**Model 1: Aggregate Factor Effects**

Mixed-effects linear regression with authority random intercepts:
```
log_fine_i = β₀ + β₁(agg_count_i) + β₂(mit_count_i) + β₃(neutral_count_i)
           + γX_i + u_a[i] + ε_i
```

**Model 2: Factor Decomposition**

Separate estimation for each Article 83(2) factor to identify differential effects.

**Model 3: Systematicity Analysis**

Construction of authority-level systematicity index:
```
Systematicity_a = Coverage_a × Consistency_a × Coherence_a
```

Where coverage measures factor discussion completeness, consistency measures directional stability, and coherence measures alignment between stated factors and outcomes.

**Model 4: Cross-Border Matching**

Nearest-neighbor matching within article cohorts using Mahalanobis distance on defendant characteristics, with matched-pair disparity tests.

**Model 5: Variance Decomposition**

Three-level mixed model partitioning variance into country, authority (nested), and case components.

---

## 5. Results

### 5.1 Descriptive Statistics

Table 1 summarizes sample characteristics by country. The sample spans 18 jurisdictions, with Italy (n=167) and Spain (n=112) contributing the most decisions. Mean fine amount varies from EUR 50 (Estonia's single case) to EUR 80.6 million (Ireland), with an overall mean of EUR 2.67 million and median of EUR 20,982.

**Table 1: Sample Characteristics by Country**

| Country | N | Mean Fine (EUR) | Median Fine (EUR) | Mean Factors Discussed |
|---------|---|-----------------|-------------------|------------------------|
| Italy | 167 | 240,544 | 20,416 | 6.7 |
| Spain | 112 | 123,138 | 9,928 | 3.6 |
| Greece | 47 | 253,487 | 10,208 | 4.2 |
| France | 38 | 10,373,201 | 497,603 | 5.2 |
| Hungary | 29 | 41,889 | 8,616 | 6.0 |
| Belgium | 28 | 23,991 | 11,219 | 5.1 |
| Germany | 20 | 6,409,075 | 451,870 | 4.8 |
| Ireland | 10 | 80,633,062 | 126,568 | 2.9 |
| Other (10) | 77 | Various | Various | Various |
| **Total** | **528** | **2,668,131** | **20,982** | **5.2** |

Factor usage varies substantially across authorities. Nature/gravity/duration is discussed in 91% of decisions, while codes/certification appears in only 2%. Mean factors discussed per decision is 5.2 of 11 possible.

### 5.2 Factor Predictiveness (H1)

Table 2 presents Model 1 results. Aggravating factor count significantly predicts higher fines (β = 0.221, p < 0.0001), with each additional factor increasing expected fine by approximately 25% (exp(0.221) = 1.247). Mitigating factor count shows the expected negative direction but lacks statistical significance (β = -0.036, p = 0.437).

**Table 2: Aggregate Factor Effects on Log Fine**

| Variable | β | SE | 95% CI | p |
|----------|---|-----|--------|---|
| Aggravating count | 0.221*** | 0.050 | [0.123, 0.319] | <0.001 |
| Mitigating count | -0.036 | 0.046 | [-0.125, 0.054] | 0.437 |
| Neutral count | 0.107 | 0.072 | [-0.034, 0.248] | 0.135 |
| Private sector | 0.403* | 0.173 | [0.064, 0.742] | 0.020 |
| Large enterprise | 1.963*** | 0.175 | [1.621, 2.305] | <0.001 |
| Complaint origin | -0.477* | 0.187 | [-0.844, -0.110] | 0.011 |
| Cross-border (OSS) | 0.996** | 0.385 | [0.241, 1.750] | 0.010 |

*Notes: N = 528. Authority random intercepts included. *** p<0.001, ** p<0.01, * p<0.05*

The asymmetry between aggravating and mitigating factor effects is notable. While both coefficients have the expected sign, only aggravating factors achieve statistical significance. This suggests DPAs more reliably increase penalties for aggravating circumstances than decrease them for mitigating ones.

Factor decomposition (Model 2) reveals heterogeneous effects across specific factors:

**Table 3: Factor-by-Factor Decomposition**

| Factor | β | p | Interpretation |
|--------|---|---|----------------|
| Data Categories Affected | +0.548*** | <0.001 | Strongest aggravating effect |
| Previous Infringements | +0.521*** | <0.001 | Strong recidivism penalty |
| Mitigation Actions | -0.335* | 0.014 | Only significant mitigator |
| Tech/Org Measures | +0.316* | 0.017 | Pre-violation negligence |
| Other Factors | +0.249* | 0.016 | Catch-all aggravation |
| Intentional/Negligent | +0.240* | 0.016 | Mental state matters |
| Cooperation | -0.064 | 0.562 | Not significant |

### 5.3 Systematicity Analysis (H2)

We compute systematicity indices for 22 authorities with at least 10 decisions. Index values range from 0.000 to 0.252, with CNIL (France) scoring highest. Regressing authority-level fine predictability (R²) on systematicity yields:

β = 0.354 (p = 0.827), r = 0.085

This null finding provides no support for H2. More systematic factor articulation does not translate to more predictable outcomes. This suggests factor discussion may function as post-hoc justification rather than determinative reasoning.

### 5.4 Cross-Border Disparities (H3)

We match 238 pairs across 40 article cohorts. Matched-pair disparity tests strongly support H3:

| Metric | Full Sample | Excl. Mega-Fines (>€10M) |
|--------|-------------|---------------------------|
| Matched pairs | 238 | 220 |
| Mean Δ log fine | 2.566*** | 2.168*** |
| t-statistic | 18.9 | 17.6 |
| Cohen's d | 1.226 | 1.392 |
| Implied ratio | 13.0x | 8.7x |

The mean log difference of 2.57 implies approximately 13-fold average fine differences (exp(2.566) = 13.0) for cases with identical article violations matched on defendant characteristics. When excluding matched pairs involving mega-fines (>€10M), the disparity remains highly significant but drops to approximately 8.7x (−33%), indicating that extreme penalties contribute substantially to cross-border variation while the core finding of inconsistency remains robust.

**Table 4: Cross-Border Gaps by Article Cohort (Selected)**

| Cohort | Pairs | Mean Δ Log Fine | Implied Ratio |
|--------|-------|-----------------|---------------|
| Art. 5 violations | 38 | 2.76 | 15.8x |
| Art. 5+6 combined | 24 | 2.76 | 15.8x |
| Art. 5+32 combined | 31 | 2.60 | 13.5x |
| Art. 32 alone | 19 | 2.09 | 8.1x |
| All matched pairs | 238 | 2.57 | 13.0x |

### 5.5 Variance Decomposition (H4)

Three-level variance decomposition supports H4:

**Table 5: Variance Partition**

| Component | Full Sample | Excl. >€10M Fines |
|-----------|-------------|-------------------|
| Between-Country | 35.7% | 34.6% |
| Between-Authority | 24.8% | **12.7%** |
| Within-Authority | 39.5% | 52.7% |
| **ICC (combined)** | **0.605** | **0.428** |

Country and authority effects jointly explain 60.5% of fine variance. The "which DPA you get" effect (24.8%) is nearly as large as national policy differences (35.7%), indicating substantial within-country enforcement heterogeneity.

**Outlier Sensitivity:** When excluding mega-fines (>€10M), authority variance drops from 24.8% to 12.7% (−12 percentage points), while country variance remains stable. This indicates that extreme penalties (like the €530M Meta fine) contribute disproportionately to authority heterogeneity. The ICC drops from 0.605 to 0.428, though it remains substantial.

### 5.6 Robustness Checks

**Specification Curve:** Analysis across 270 specifications (expanded from 108) yields mean β = 0.214 for aggravating count, with all estimates positive, all significant at p < 0.05, and 86% significant at p < 0.01.

**Bootstrap CIs:** 1,000-replicate bootstrap produces 95% CI [0.127, 0.395] for aggravating count, clearly excluding zero.

**Leave-One-Country-Out:** All coefficients remain significant (p < 0.001) when any country is excluded. Maximum coefficient change is 23.4% (Ireland), but all estimates remain in the [0.23, 0.27] range.

**Placebo Tests:** Shuffling factors within countries produces placebo mean β = 0.019, with original estimate in the 100th percentile of the null distribution (p < 0.0001).

**Mega-Fine Sensitivity:** The fine distribution is highly skewed (15 fines >€10M account for 93% of total fine value). Excluding mega-fines, the aggravating factor coefficient remains stable (β = 0.24, p < 0.0001, −8% change). Excluding fines >€1M, β = 0.21 (−18% change), still highly significant.

**Winsorization:** Capping fines at various percentiles (99th, 97.5th, 95th, 90th) produces stable coefficients ranging from 0.21 to 0.26, all highly significant. Notably, winsorization reduces the authority-level ICC from 0.57 to 0.41, confirming that mega-fines contribute disproportionately to enforcement heterogeneity.

---

## 6. Discussion

### 6.1 Summary

Our analysis provides systematic evidence on GDPR enforcement patterns:

1. **Partial factor predictiveness (H1):** Aggravating factors significantly increase fines (~25% per factor), but mitigating factors lack symmetric effect. This finding is robust to outlier exclusion and winsorization.

2. **No systematicity benefit (H2):** More complete factor discussion does not improve outcome predictability.

3. **Substantial cross-border disparities (H3):** Matched cases show 13x average fine differences across jurisdictions (8.7x when excluding mega-fine pairs). Core finding robust but magnitude sensitive to outliers.

4. **Pronounced authority heterogeneity (H4):** "Which DPA you get" explains 25% of variance beyond country effects (13% excluding mega-fines). Extreme penalties contribute disproportionately to authority-level variation.

### 6.2 Implications

**For Regulatory Theory:** The asymmetric factor application challenges rational penalty models. If factors genuinely determine outcomes, mitigating circumstances should reduce penalties symmetrically with aggravating circumstances increasing them. The observed asymmetry suggests either (1) systematic underdocumentation of mitigation, (2) institutional incentives favoring penalty enhancement, or (3) post-hoc reasoning that selects supporting factors rather than weighing all considerations.

**For Legal Certainty:** The substantial authority-level variance creates enforcement lottery dynamics. Organizations face similar violations but dramatically different penalty risks depending on which DPA investigates. This unpredictability complicates compliance cost-benefit analysis and may encourage regulatory arbitrage.

**For Harmonization Policy:** The 13x average cross-border gap suggests current coordination mechanisms (EDPB consistency mechanism, OSS) are insufficient to ensure comparable outcomes. More prescriptive approaches—such as quantitative sentencing guidelines common in competition law—may be needed.

### 6.3 Limitations

Several limitations warrant caution:

1. **Selection bias:** Only publicly disclosed decisions; informal resolutions not observed
2. **Measurement:** AI-extracted coding may introduce noise, though schema validation reduces this concern
3. **Temporal scope:** 2018-2024 period may not represent mature enforcement
4. **Causation:** Observational design limits causal claims despite rich controls
5. **Sample size:** Authority-level analyses limited by small n for some DPAs
6. **Outlier sensitivity:** The highly skewed fine distribution (93% of total fines from 2.8% of cases) means magnitude claims should be interpreted with caution. Cross-border disparity and authority heterogeneity estimates are sensitive to mega-fine inclusion, though core findings remain robust.

---

## 7. Conclusion

This paper provides the first systematic quantitative analysis of GDPR penalty factor application and cross-border consistency. Our findings reveal substantial enforcement heterogeneity that challenges the regulation's harmonization ambitions. Aggravating factors reliably increase penalties, but mitigating factors do not symmetrically decrease them. Matched cases show 13-fold average fine differences across jurisdictions (8.7-fold excluding mega-fines), and "which DPA you get" explains nearly as much variance as national policy differences (though this effect is substantially driven by extreme penalties).

These findings have implications for regulatory design (factor weighting guidelines), compliance strategy (jurisdictional risk assessment), and ongoing debates about European enforcement harmonization. Future research should examine temporal dynamics, appellate outcomes, and extensions to other regulatory domains.

---

## References

Adshead, S., Sheridan, R., & Sheridan, A. (2021). GDPR: Two years on. *Journal of Data Protection & Privacy*, 4(2), 108-128.

Becker, G. S. (1968). Crime and punishment: An economic approach. *Journal of Political Economy*, 76(2), 169-217.

Custers, B., Dechesne, F., Sears, A. M., Tani, T., & van der Hof, S. (2022). A comparison of data protection legislation and policies across the EU. *Computer Law & Security Review*, 34(2), 234-243.

European Parliament. (2021). *GDPR Implementation: State of Play*. Policy Department for Citizens' Rights and Constitutional Affairs.

Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. *Nature Human Behaviour*, 4(11), 1208-1214.

Stigler, G. J. (1970). The optimum enforcement of laws. *Journal of Political Economy*, 78(3), 526-536.

Stuart, E. A. (2010). Matching methods for causal inference: A review and a look forward. *Statistical Science*, 25(1), 1-21.

---

## Appendix A: Model Diagnostics

### A.1 Model 1 Specification

```
log_fine_2025 ~ art83_aggravating_count + art83_mitigating_count
              + art83_neutral_count + is_private + is_large_enterprise
              + complaint_origin + audit_origin + oss_case
              + breach_has_art5 + breach_has_art6 + breach_has_art32
              + breach_has_art33 + has_cctv + has_marketing
              + (1 | authority)
```

**Model Fit:**
- N = 528 observations
- N = 66 authority clusters
- R² (marginal) = 0.601
- R² (conditional) = 0.671
- ICC = 0.585
- Log-likelihood = -1,010.74

### A.2 Matching Quality

Covariate balance after matching:
- Standardized mean differences < 0.25 for all matching variables
- Variance ratios within [0.5, 2.0] range
- No unmatched cases within eligible cohorts

---

*Word count: ~3,500*
*Tables: 5*
*Figures: Referenced but not embedded*
