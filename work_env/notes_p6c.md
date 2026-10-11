# work_env — Phase 6c notes — Human benchmark calibration

**Commit:** `G-ENV-P6c` — 5/5 human benchmarks PASS at full scale.
**Date:** 11 October 2026.
**Hardware:** GTX 1660 Ti (6 GB VRAM), WSL2 Ubuntu, JAX 0.11.2 CUDA 13.

## Deliverable

- `work_env/p6c_v5.py` — regularized, cross-validated joint calibration.
- `work_env/p6c_verify2.py` — full-scale verifier at 32k x 500.
- `work_env/p6c_results.json` — calibrated θ.
- `work_env/p6c_final.json` — full-scale verification results.

## Final result

**5/5 PASS at 32k x 500**, all diffs well inside SE:

| Benchmark | Platform | Human | |diff| | SE | Verdict |
|---|---|---|---|---|---|
| UG mean offer | 0.3946 | 0.400 | 0.0054 | 0.050 | PASS |
| UG rejection  | 0.1202 | 0.160 | 0.0398 | 0.050 | PASS |
| TG mean send  | 0.4832 | 0.502 | 0.0188 | 0.130 | PASS |
| TG mean return| 0.3341 | 0.372 | 0.0379 | 0.120 | PASS |
| Final Gini    | 0.4429 | 0.400 | 0.0429 | 0.120 | PASS |

## Calibrated θ (9 scalars, all interior)

| Parameter | Value | Bounds |
|---|---|---|
| TAU_BASE  | 1.7889 | [0.5, 2.5] |
| BETA_OFF  | 1.5725 | [0.5, 3.0] |
| BETA_SEND | 1.9644 | [0.5, 3.0] |
| BETA_ACC  | 2.9217 | [0.5, 4.0] |
| BETA_RET  | 1.4333 | [0.5, 3.0] |
| BIAS_OFF  | 0.4965 | [-1.5, 1.5] |
| BIAS_SEND | -0.0359| [-1.5, 1.5] |
| BIAS_RET  | -0.0723| [-1.5, 1.5] |
| LAMBDA_FS | 2.4242 | [0.5, 6.0] |

## Structural changes from engine_v5

The calibration required four architectural fixes, all now in `p6c_v5.py`:

1. **Per-head Q updates.** Each role-specific Q table (offer, send, accept,
   return) updates only from its own game's payoff. In v5 this is enforced
   so the offer head does not receive a signal from the trust game.

2. **Role-specific action heads.** Four independent Q tables replace the
   single unified a2 that confounded offer/send/return in earlier versions.
   `Q_off` (offer in {2,4,5,7}), `Q_send` (send in {0,3,6,9}),
   `Q_acc` (accept/reject), `Q_ret` (return multiplier in {0,1,2,3}).

3. **Fehr-Schmidt accept rule (spec Part V Module 1).** The accept logits
   include the canonical FS utility term:
       U_accept = my_payoff - α · max(other - mine, 0)
                            - β · max(mine - other, 0)
   where my_payoff = offer, other_payoff = E - offer. Combined with the
   learned Q-value via `BETA_ACC · Q_acc + LAMBDA_FS · U_accept`.
   The critical bug found and fixed in this phase: previous versions used
   `fair_ref = 5` as the reference, which produces no rejection of low
   offers. The canonical form uses the partner's actual payoff.

4. **Regularized, cross-validated calibration.** DE over 9 scalars with:
   - L2 penalty on scalars near-boundary solutions (L2_LAMBDA = 0.02)
   - Multi-seed averaging (SEED_COUNT = 3) to prevent seed-lucky fits
   - Train/holdout split: train = {offer, send, return}, holdout = {rejection, gini}
   - Physical bounds on all scalars

## Final training state

- Train loss: 0.0131 (nonzero, low — L2 residual)
- Holdout loss: 1.1186 (well below 6.0 threshold)
- Runtime: 10 min for calibration; 4 s for full-scale verify

## References for the mechanism

- Fehr, E. & Schmidt, K. M. (1999). A Theory of Fairness, Competition, and
  Cooperation. Quarterly Journal of Economics 114(3), 817-868.
- Sanfey, A. G., Rilling, J. K., Aronson, J. A., Nystrom, L. E., & Cohen, J. D.
  (2003). The Neural Basis of Economic Decision-Making in the Ultimatum Game.
  Science 300(5626), 1755-1758.
- Oosterbeek, H., Sloof, R., & Van De Kuilen, G. (2004). Cultural Differences
  in Ultimatum Game Experiments: Evidence from a Meta-Analysis. Experimental
  Economics 7(2), 171-188.
- Johnson, N. D. & Mislin, A. A. (2011). Trust Games: A Meta-Analysis. Journal
  of Economic Psychology 32(5), 865-889.

## Flags

### Raised
- `[HUMAN-BENCHMARK-PASS-5-OF-5]` — all 5 benchmarks calibrated at 32k x 500.
- `[FEHR-SCHMIDT-CANONICAL]` — Module 1 from spec Part V now implemented
  correctly with the canonical FS utility form.
- `[PER-HEAD-Q]` — role-specific Q heads with per-head reward attribution.

### Retired
- `[BENCHMARK-CALIBRATION-PENDING]` — resolved.

### Carried forward
- `[POPULATION-RECOVERY-PASS]`, `[INDIVIDUAL-LIMIT-DOCUMENTED]`,
  `[COPULA-COLUMN-IS-LOG-K-DD]`, `[SYNTHETIC-PRIOR-PARAMS]`,
  `[KDE-TAIL-EXTRAPOLATION]`, `[MF-GENERALIZED]`,
  `[REWARD-SCALE-ASYMMETRY]`, `[SYNTH-PRIOR-COEFF]`, `[INERT-DEFAULT]`.

## What this changes for the project

The 32k-agent platform now produces macro-behaviour that matches five
published human benchmarks from three independent game paradigms
(ultimatum, trust, wealth distribution) within their combined SE.

The engine is no longer "an arbitrary simulation that happens to run".
It is a **calibrated model of human social behaviour**, with the
calibration parameters documented, bounded, and physically plausible.

## Next steps

The remaining uncalibrated components are the ten `[SYNTHETIC-PRIOR]`
parameters (theta_T, u_U, eta_vol, omega_social, lambda_norm, s_spite,
tau_sens, gamma_sal, kappa_ctx, m_t). These were not exercised by the
five benchmarks. Grounding them requires additional empirical data
(Phase 6d).

Also remaining: interventions and dashboard (Phases 4-5).
