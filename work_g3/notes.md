# G3 — SD3 bifactor CFA — notes

**Input:** `work_sd3/sd3_items_revkeyed.csv` (17,740 subjects × 27 items)
**Estimator:** semopy maximum likelihood (continuous treatment of 5-point Likert).
**Models compared:** three correlated factors (A), unidimensional (B), bifactor with orthogonal general + three specifics (C).

## Fit comparison

| Model | chi2 | df | CFI | TLI | RMSEA |
| :--- | ---: | ---: | ---: | ---: | ---: |
| A: 3 correlated factors | 26228.80 | 321 | 0.848 | 0.834 | 0.0675 |
| B: unidimensional | 38914.03 | 324 | 0.773 | 0.755 | 0.0819 |
| **C: bifactor (orthogonal)** | **17175.09** | **297** | **0.901** | **0.883** | **0.0566** |

Delta-chi2, A vs C: 9053.7 on 24 df. p < 1e-100.
CFI improvement A -> C: **+0.053** (decision threshold: 0.02).

## G3 decision

**PASS — bifactor structure replicates.**

The bifactor model outperforms the three-correlated-factor model by
substantially more than the pre-declared minimum (CFI +0.053 vs. 0.02;
RMSEA 0.0566 vs. 0.0675; TLI +0.049). The general factor is not
statistical noise — it explains systematic covariance across all 27 items
that the three correlated factors cannot absorb.

This result is consistent with the SD3 meta-analytic literature
(Knitter et al., 2025) reporting bifactor superiority for the SD3 in
large samples.

## Caveats and limitations

1. **Continuous-ML estimator on categorical data.** The 27 items are
   5-point Likert responses. The strictly correct approach for ordinal
   indicators is DWLS or WLSMV. semopy's ML-CFA treats the items as
   continuous, which can slightly inflate chi2 and understate CFI/RMSEA
   fit. This does not change the *direction* of the A-vs-C comparison —
   bifactor still wins by a wide margin — but the exact fit indices
   should be treated as approximations. Flag: `[CONTINUOUS-ML-CFA]`.

2. **BIC/AIC not usable.** semopy reported AIC/BIC on a per-observation
   scale that does not correspond to the standard chi2-based definition.
   These were not used for model selection.

3. **chi2 p-value is trivially zero** at N = 17,740 for any model.
   Model selection relies on CFI / TLI / RMSEA / delta-chi2.

4. **No omega-hierarchical / ECV computed.** These are the
   recommended follow-up to quantify how much variance the general
   factor absorbs. Queue for a follow-up if downstream analyses depend
   on that split.

## Parameter dictionary status

- **D (general Dark Core)**: bifactor-replicated. Loadings from the
  general factor onto all 27 items are saved in
  `sd3_bifactor_loadings.csv`. These become the item weights for the
  composite D score used by the Vine Copula.
- **S_M, S_P, S_N**: specific factors identified after partialling out
  D. Loadings also in the loadings CSV.
- Flags: `[HCP-OPEN-PSYCHOMETRICS]` (not HCP), `[CONTINUOUS-ML-CFA]`.

## Reproducibility

Script: `cfa_sd3.py`. Log: `cfa_sd3.log`. Runtime: ~1 s.
cd work_g3 && python cfa_sd3.py
