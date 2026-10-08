# work_env — Phase 5 notes — KDE resample for kde_anchored columns

**Commit:** `G-ENV-P5` — continuous two-step parameters, 149-arch limitation resolved.
**Date:** 08 October 2026.
**Hardware:** GTX 1660 Ti (6 GB VRAM), WSL2 Ubuntu, JAX 0.11.2 CUDA 13.

## Deliverable

- `work_g2/patch_kde_sampling.py` — KDE resample for the 6 kde_anchored columns.
- `work_g2/synthetic_32k.pkl` updated (backup at `synthetic_32k.pkl.pre_kdefix`).
- New work_env baseline hash `e87e5ad5...`.

## Problem

`marginal_models.csv` labeled 6 columns `kde_anchored` — alpha_R, beta_choice,
phi_per, w_MB, a_inhib, k_DD. But the G2 builder sampled them via rank-matching
on the raw source arrays. When source N (149, 264, 476) < target N (32000),
the synthetic column had exactly N_source uniques. Effect: 32000 agents
collapsed onto 149 cognitive archetypes.

## Fix

For each of the 6 columns:
1. Fit a Gaussian KDE to the transformed source (logit for [0,1],
   log for positive, identity otherwise).
2. Draw exactly n=32000 continuous samples from the KDE.
3. Sort ascending, invert the transform.
4. Rank-match: place the sorted new values into the rank positions of
   the original column.

Because step 4 only permutes the marginal values within their existing
rank positions, the Spearman correlation between this column and every
other column is **exactly preserved**. Copula structure unchanged.

## Result

| Column | src_n | uniq_before | uniq_after | src_range | new_range |
|---|---|---|---|---|---|
| alpha_R | 149 | 149 | 32000 | [0.0000, 0.9983] | [0.0000, 0.9998] |
| beta_choice | 149 | 149 | 32000 | [0.7491, 29.5330] | [0.3746, 59.8493] |
| phi_per | 149 | 149 | 32000 | [-0.1547, 1.8184] | [-0.4905, 2.1089] |
| w_MB | 149 | 149 | 32000 | [0.0000, 1.0000] | [0.0000, 1.0000] |
| a_inhib | 264 | 264 | 32000 | [0.1113, 0.9437] | [0.0832, 1.2225] |
| k_DD | 476 | 26 | 32000 | [-9.1150, 3.1781] | [-11.1597, 4.6299] |

Range expansion (KDE tail extrapolation beyond the source min/max) is
expected and statistically correct: KDE with Silverman bandwidth produces
a continuous density over the entire real line, and 32000 draws are more
likely to land in the tails than the 149 source observations. All
expansions are bounded by the natural transform ranges:
- w_MB clipped to [0, 1] by logit inverse (no violation).
- alpha_R max 0.9998 < 1 (fine).
- beta_choice max 59.85 — very sharp softmax for a handful of agents.
- k_DD min -11.16 → k ≈ 1.6e-5 → γ ≈ 0.99998 — very patient agents.
- a_inhib max 1.22 — outside source range but log-transform extrapolation,
  acceptable for a scale parameter.

## New baseline shift

| Metric | P4 | P5 | Delta |
|---|---|---|---|
| state_hash | 2c238aca... | **e87e5ad5...** | (expected) |
| coop | 0.9497 | 0.9533 | +0.004 |
| gini | 0.3960 | 0.4044 | +0.008 |
| distress | 0.4652 | 0.4653 | ~0 |
| w_mb_eff | 0.4569 | 0.4463 | −0.011 |
| defection | 60 | 57 | −3 |
| Q_net | 30.07 | 29.63 | −0.44 |
| reward | 2.1314 | 2.2035 | +0.07 |
| mean_a2 | 2.0635 | 2.0711 | +0.008 |

Agent 100 (traced): interacting rate 0.060 → 0.082. Continuous parameter
draws give some agents slightly more extreme engagement propensity.

## Trace re-verification

`test_xai_trace.py` re-run on new baseline:
- Reference hash: `e87e5ad5...`
- Trace agent 100: hash == reference. PASS
- Trace agent 0: hash == reference. PASS
- Trace shape: (450, 24). Wall: 2.0s.
- All checkpoint replays match.

## Flags

### Retired
- `[149-ARCHETYPES]` — resolved. The four two-step parameters now have
  continuous distributions over 32000 distinct values.

### Raised
- `[KDE-TAIL-EXTRAPOLATION]` — for the 6 kde_anchored columns, 32000 KDE
  draws extend past the source min/max. This is statistically expected
  but must be documented. Any downstream claim involving extreme values
  (e.g., "the most patient agent has gamma = 0.99998") must acknowledge
  that this value is a KDE extrapolation, not a directly observed subject.

### Carried forward
- `[COPULA-COLUMN-IS-LOG-K-DD]`, `[SYNTHETIC-PRIOR-PARAMS]`,
  `[MF-GENERALIZED]`, `[REWARD-SCALE-ASYMMETRY]`, `[SYNTH-PRIOR-COEFF]`,
  `[INERT-DEFAULT]`.

## Remaining audit items

1. **Coupled-engine recovery** — never run. Must run before any scientific
   claim. This is the last open item from the audit.
2. **Human benchmark validation** — deferred.
3. **Phase 6 — interventions and dashboard** — deferred until 1 and 2.

## Out of scope (Phase 5)

No interventions. No dashboard. Copula fix only.
