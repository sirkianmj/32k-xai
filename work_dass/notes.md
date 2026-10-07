# DASS extraction — notes

**Source:** `Data_Xai32kProje/XAI_Mental_Health_DASS/data.csv`
**Platform:** Open Psychometrics (same platform as IPIP-FFM). No participant
ID; cannot be linked row-by-row to HCP or to the IPIP sample.
**Collection window:** 2017–2019. Self-selected online sample.

## Deviation from the Master Spec

The Master Technical Specification assumes **DASS-21** (7 items per subscale).
The file on disk contains the **full DASS-42** (14 items per subscale). This is
a documented deviation. Both scales have been extracted:

- `dass42_subscales.csv` — 42-item scores, range [0, 42] per subscale
- `dass21_subscales.csv` — 21-item subset, range [0, 21] per subscale

The DASS-21 subset uses the standard Lovibond & Lovibond (1995) mapping. The
two scales are highly correlated and both have acceptable reliability
(see below). The G4 bifactor CFA can be run on either; the specification's
"general distress factor" is expected to appear in both.

## Quality filters applied

1. NaN filter (none triggered — 0 missing items).
2. Range filter: all 42 items must be integers in {1, 2, 3, 4} (DASS Likert).
   All 39,775 rows passed.
3. Optional `uniquenetworklocation == 1` subset (31,819 rows) removes
   submissions that shared a network with other submissions. Available via
   the `uniquenetworklocation` column in both CSVs; not applied by default.

## Scoring

Raw item responses 1–4 were converted to 0–3 (standard DASS scoring) before
summing. Normalized columns `*_n` divide by 42 (DASS-42) or 21 (DASS-21) to
land in [0, 1].

## Reliability (Cronbach's alpha)

| Scale | Depression | Anxiety | Stress |
| :--- | :--- | :--- | :--- |
| DASS-42 | 0.9561 | 0.9163 | 0.9284 |
| DASS-21 | 0.9205 | 0.8323 | 0.8584 |

All subscales exceed the 0.80 threshold for research use; both scales match
published reliability estimates.

## Sample characteristics (caution)

This is a **self-selected, distress-enriched sample** (people take a DASS
because they suspect they are distressed). Observed means:

- DASS-42: Dep = 21.06, Anx = 16.06, Str = 21.15
- DASS-21: Dep = 10.51, Anx =  7.81, Str = 10.93

These are substantially above normative community means. **Implication for
the Vine Copula:** marginals derived from this sample represent the *shape*
of the distribution (skew, tail behavior, subscale covariance), not the
population location. When the copula is fitted, use these marginals for
shape; do not treat the mean as a general-population anchor.

## Subscale correlations

| Scale | Dep-Anx | Dep-Str | Anx-Str |
| :--- | :--- | :--- | :--- |
| DASS-42 | 0.670 | 0.740 | 0.802 |
| DASS-21 | 0.620 | 0.693 | 0.727 |

All three subscales are positively correlated at r > 0.6 — consistent with
the spec's commitment to a bifactor model (general distress + 3 specifics),
not three orthogonal factors.

## Reproducibility

Script: `score_dass.py`. Log: `score_dass.log`. Runtime: 0.4 s. Run with:
cd work_dass && python score_dass.py
