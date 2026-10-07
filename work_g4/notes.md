# G4 — DASS bifactor CFA — notes

**Input:** `work_g4/dass42_items.csv` (39,775 subjects × 42 items, 0–3 scored)
**Source:** Open Psychometrics DASS survey (same sample as `work_dass/`)
**Estimator:** semopy maximum likelihood (continuous treatment of ordinal items)
**Models:** (A) three correlated factors; (B) unidimensional; (C) bifactor with orthogonal general + three specifics.

## Fit comparison

| Model | chi2 | df | CFI | TLI | RMSEA |
| :--- | ---: | ---: | ---: | ---: | ---: |
| A: 3 correlated (Dep/Anx/Str) | 112410.92 | 816 | 0.898 | 0.892 | 0.0586 |
| B: unidimensional | 246909.85 | 819 | 0.775 | 0.763 | 0.0869 |
| **C: bifactor (G + 3 specifics)** | **75616.02** | **777** | **0.931** | **0.924** | **0.0492** |

Delta-chi2, A vs C: 36794.9 on 39 df. p < 1e-100.
CFI improvement A -> C: **+0.033** (decision threshold: 0.02).
RMSEA improvement: 0.0586 -> 0.0492.

## G4 decision

**PASS — bifactor structure replicates.**

The bifactor model beats the three-correlated-factor model by more than the pre-declared threshold (CFI +0.033 > 0.02; RMSEA improvement +0.009). All three subscales load substantially on a general factor while retaining meaningful specific-factor variance. This matches the cross-cultural bifactor result on the DASS-21 (Zanon et al., 2020) and the spec's Part II.2 commitment to a bifactor DASS.

## Implications for the specification

The spec calls for `w_MB` dynamic decay under stress to be driven by the **general distress factor**, not three orthogonal factors. This CFA confirms the general factor is empirically warranted:

- All 42 items have a primary loading on G, not just on their subscale.
- The specific-factor residuals for depression, anxiety, and stress are present and non-trivial but not dominant.

Consequence: **general distress** is the correct compound for the stress-modulator input to `w_MB`. The subscale-specific residuals become secondary modulators per the spec's Part II.2, and must carry lower reliability flags.

## Caveats

1. `[CONTINUOUS-ML-CFA]` — same as G3. Items are ordinal; proper treatment is DWLS/WLSMV. The direction of the A-vs-C result is robust to this choice; the absolute fit indices are approximations.
2. `[DISTRESS-ENRICHED]` — the sample is self-selected and distress-enriched (DASS-42 Dep mean 21.1 / 42). Fit statistics are unaffected by the mean shift, but downstream Vine Copula marginals must treat this sample as shape-only (already flagged in `work_dass/notes.md`).
3. **No omega-hierarchical / ECV.** The recommended next step if downstream analyses require quantifying the variance split between G and the specifics.

## Parameter dictionary status

- **General distress** (modulator of `w_MB`): bifactor-replicated. Loadings saved in `dass_bifactor_loadings.csv` (136 rows). Weights for the composite G score come from this file.
- **Subscale specifics (Dep / Anx / Str)**: identified after partialling out G. Loadings in same file. Reliability flag: `[SECONDARY-MODULATOR]`.
- Flags: `[HCP-OPEN-PSYCHOMETRICS]`, `[CONTINUOUS-ML-CFA]`, `[DISTRESS-ENRICHED]`.

## Reproducibility

Script: `cfa_dass.py`. Log: `cfa_dass.log`. Runtime: ~1 s.
cd work_g4 && python cfa_dass.py
