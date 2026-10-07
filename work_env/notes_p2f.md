# work_env — Phase 2f notes — deep agent scale invariance

**Commit:** `G-ENV-P2f` — engine_v5 verified scale-invariant 32k → 2M.
**Date:** 07 October 2026.
**Hardware:** GTX 1660 Ti (6 GB VRAM), WSL2 Ubuntu, JAX 0.11.2 CUDA 13.

## Deliverable

- `scale_test_v5.py` — per-N diagnostic that builds a WS graph, resamples
  copula, and runs 500 ticks of engine_v5.
- `scale_v5.log`, `scale_v5_results.json` — logs and per-N macro curves.

## Result

| N | wall_s | VRAM_MB | coop | gini | distress | mean_a2 | reward |
|---|---|---|---|---|---|---|---|
| 32,000 | 1.97 | 13.8 | 0.9611 | 0.5627 | 0.4638 | 2.1110 | 0.5014 |
| 100,000 | 2.99 | 42.7 | 0.9737 | 0.5631 | 0.4632 | 2.1119 | 0.4895 |
| 500,000 | 5.76 | 206.1 | 0.9623 | 0.5629 | 0.4629 | 2.1116 | 0.4947 |
| 1,000,000 | 9.61 | 408.3 | 0.9596 | 0.5624 | 0.4629 | 2.1100 | 0.4823 |
| 2,000,000 | 18.11 | 864.5 | 0.9610 | 0.5624 | 0.4628 | 2.1096 | 0.4845 |

### Scale-invariance vs. N=32k (per-tick RMSE)

| N | RMSE_coop | RMSE_gini | RMSE_a2 |
|---|---|---|---|
| 100k | 0.01407 | 0.00113 | 0.00473 |
| 500k | 0.01283 | 0.00121 | 0.00488 |
| 1M | 0.01289 | 0.00140 | 0.00478 |
| 2M | 0.01276 | 0.00149 | 0.00491 |

**Verdict: scale-invariant, deep agent, 32k → 2M.**
- Final coop varies by ≤1.4 pp across 62× range in N.
- Final gini varies by ≤0.0007 across all N.
- Final mean_a2 varies by ≤0.0023 across all N.
- RMSE converges and plateaus at bootstrap noise floor (identical to P2b1).
- No qualitative change in equilibrium with N.
- No OOM at 2M. VRAM at 2M: 864 MB — 16% of 5.1 GB usable.

## Findings

### Finding 1 — Deep agent preserves scale-invariance
The deeper dynamics (P_hat learning, trust game, Fehr-Schmidt,
30-parameter copula) do not introduce N-dependence. The same equilibrium
holds at every scale. 32k is scientifically representative for the
phenomena this platform studies.

### Finding 2 — Fair-strategy convergence is N-independent
mean_a2 = 2.11 across all N. The population converges on the "fair"
action category (ulti offer 5, trust send 6) regardless of scale.

### Finding 3 — Reward magnitude slightly decreases with N
- 32k: reward 0.5014
- 2M: reward 0.4845
Difference = 3.4% decrease. Likely a finite-size effect on match rate
(larger N → more uniform random partner choices → slightly lower variance
in mutual-pick success). Not scientifically significant at current
resolution but worth a Phase 6 sweep at N = 100k with different seeds.

### Finding 4 — Wall time and VRAM scale linearly
- VRAM: ~430 bytes/agent (state + XLA temporaries at deep agent level).
- Wall: ~9 µs/agent/tick at 1M–2M.
- 5.1 GB usable VRAM would support ~12M agents in one process, but
  XLA peak temporaries at 12M would be ~5 GB — near the ceiling.
  Realistic single-process max is ~8–10M agents.

## Flags carried forward

- `[NONINTERACTING-Q-DRIFT]` — Q updates fire even when non-interacting.
- `[REWARD-SCALE-ASYMMETRY]` — trust payoffs ~8× ultimatum.
- `[COPULA-DEFECT-k_DD]` — G2 copula k_DD violates spec >0 constraint.
- `[TANH-GAMMA-PROVISIONAL]` — γ transform is a stopgap.
- `[MF-GENERALIZED]` — v5 MF form departs from hBayesDM ts_par7.
- `[SYNTH-PRIOR-COEFF]` — all coupling constants provisional.

## Recommendation

- Mark platform `[SCALE-INVARIANT-DEEP-32K]`.
- 32k remains the operating scale for all downstream experiments.
- No sharding required for the current scientific questions.
- Sharding design becomes necessary only if a specific experiment
  requires N > 5M (tail wealth statistics, extreme-order Gini).

## Out of scope (Phase 2f)

No sharding. No 5M+ agents. No coefficient sweeps. No XAI trace.
Phase 2g (checkpoints) is the next step.
