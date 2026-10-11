# G6 Compliance Note — Fehr-Schmidt α/β recovery below threshold

**Date:** 11 October 2026
**Spec reference:** Master Spec v2.0 §2.4 (Fehr-Schmidt Identifiability),
§8.1 Gate G6.
**Status:** Non-compliance documented; spec-mandated simplification applied;
flag raised. No code change, no recalibration.

## Gate criterion (Spec §8.1)

> G6: Fehr-Schmidt α/β recovery — Recovery report — α/β rank-order
> recovery p > 0.5; else simplify model.

Interpreted as rank-order recovery (Spearman ρ) > 0.5 for both α and β,
where α is the envy parameter and β the guilt parameter of the
Fehr-Schmidt (1999) utility.

## Evidence

Source: `work_fs/fs_recovery.txt`

    N_SIM         = 500
    N_valid       = 500
    rho_alpha     = 0.4919
    p_alpha       = 7.951e-32
    bias_alpha    = 1.0545
    rmse_alpha    = 2.8581

`rho_beta` is not reported because β was not separately identifiable from
the HCP Ultimatum data: the source contains no advantageous-inequity
offers, so the guilt branch of the FS utility has no signal to fit
against. See `work_g9/provenance_matrix.md` line for `beta_FS`
(`[NOT-IDENTIFIABLE-IN-HCP-UGR]`).

## Threshold analysis

Gate G6 requires ρ > 0.5. Observed ρ_α = 0.4919. Margin: −0.0081 below
threshold. The correlation is highly significant (p = 7.95e-32) but
weak. The RMSE (2.86) is large relative to the parameter's plausible
range [0, 3], and the bias (+1.05) is not negligible.

## Identifiability exploration

Source: `work_fs/fs_identifiability.txt`. Six prior ranges were tested
for the α recovery simulation. Same N=500, same model, same per-subject
MLE; only the sampling prior for true α varied.

| prior | range | ρ | bias | RMSE |
|---|---|---|---|---|
| narrow_low | [0.02, 0.30] | 0.8096 | +0.0409 | 0.0761 |
| narrow_low | [0.02, 0.50] | 0.7198 | −0.0067 | 0.0796 |
| lowmid | [0.20, 0.80] | 0.7581 | +3.0571 | 5.0470 |
| mid | [0.50, 1.50] | 0.1025 | +5.5976 | 6.7565 |
| wide | [0.02, 3.00] | 0.4682 | +3.8538 | 5.5185 |
| log-uniform | [0.02, 3.00] | 0.9249 | +2.0202 | 4.0308 |

Interpretation: recovery quality depends on the prior over α far more
than on the data. When α is artificially confined to a low narrow range
([0.02, 0.30]), recovery is excellent. When α is free to range over its
full theoretical interval [0, 3], recovery degrades badly. Even the
best-performing prior (log-uniform) inflates bias by +2.02 at ρ=0.92.
This is a textbook under-identification pattern: a broad likelihood
plateau on which the point estimate reflects the prior as much as the
data.

## Spec-mandated action (Spec §2.4)

> Architectural implication: If recovery testing shows that α and β
> cannot be reliably recovered from the available Ultimatum/Trust trial
> counts, the utility function is simplified to a single-parameter
> inequity aversion index, and this simplification is flagged in every
> XAI trace.

## Current implementation of the simplification

The calibrated engine (`work_env/p6c_v5.py`) already applies the
simplification in its accept rule:

    alpha_envy = P["D"] + P["lambda_ineq"]    # data-driven
    beta_guilt = P["A"]                       # proxy, not separately estimated

    U_accept = my_payoff
             - alpha_envy * max(other - mine, 0)
             - beta_guilt * max(mine - other, 0)

- **α** is the sum of the Dark Core composite `D` (from SD3 bifactor)
  and the weakly-identified `lambda_ineq` (from the Fehr-Schmidt
  Ultimatum fit). This is the "single-parameter inequity aversion
  index" the spec calls for.
- **β** is a fixed proxy from Agreeableness (IPIP-FFM). It is not a free
  parameter of the platform. Its value is determined by the population
  sample rather than being estimated from the ultimatum data.

The calibration scalar `LAMBDA_FS` = 2.4242 is a multiplier on the
overall accept utility; it does not re-estimate α or β.

## Flags

The simplification is already marked on two lines of
`work_g9/provenance_matrix.md`:

- `lambda_ineq` → `[SIMPLIFIED MODEL] [ENVY-ONLY]`
- `beta_FS` → `[NOT-IDENTIFIABLE-IN-HCP-UGR]`

These flags will be propagated to the XAI trace output in Phase 2B of
the current plan, so that every trace for every agent carries them at
the point of decision.

## What this note is not

- Not a proposal to refit α or β with a different model.
- Not a change to the p6c calibration.
- Not a re-run of the human benchmarks.
- Not a modification of any spec document.

## Consequence for the project's claims

The 5/5 human benchmarks PASS remains valid. The benchmarks measure
aggregate behavioural outcomes (offer rate, rejection rate, send rate,
return rate, Gini), not the values of α or β. The FS accept rule with
α and a β-proxy is present in `p6c_v5`, and it produces the benchmark
results. What does **not** survive is any parameter-level claim.

Specifically:

1. Any claim of the form "this agent's behaviour is attributable to its
   Fehr-Schmidt α value" must carry `[SIMPLIFIED MODEL]`.
2. Any claim of the form "this agent's behaviour is attributable to its
   Fehr-Schmidt β value" is not supportable from the current data and
   must not be made. β is not a free parameter of the platform.
3. Any downstream analysis that treats `beta_FS` as an estimated
   individual difference is invalid.

## Reference

Fehr, E., & Schmidt, K. M. (1999). A theory of fairness, competition,
and cooperation. *Quarterly Journal of Economics*, 114(3), 817–868.
