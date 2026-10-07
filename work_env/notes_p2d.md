# work_env — Phase 2d notes — true dual-system agent

**Commit:** `G-ENV-P2d` — real Model-Based Bellman + learned P_hat + γ from k_DD.
**Date:** 07 October 2026.
**Hardware:** GTX 1660 Ti (6 GB VRAM), WSL2 Ubuntu, JAX 0.11.2 CUDA 13.

## Deliverable

- `engine_v4.py` — real two-step task, per-agent learned P_hat transition tensor.
- `run_32k_v4.py` — driver (500 ticks, CSV log, final state, hash).
- `diag_v4.py` — N=1000 diagnostic with per-50 NaN/Inf monitoring.
- `run_32k_v4.log`, `final_state_v4.pkl`, `run_32k_v4_summary.txt`.

## Result

- Agents: 32,000 on WS (k=6, p=0.1), degree range [3, 10].
- Ticks: 500. Seed: 42.
- Wall (GPU): **1.796 s**.
- State hash: `9f17409c5ac26c87b1f585c16744b1a08c638158fd0680f26c28bc6fef611c27`.
- Final coop: 0.3643.
- Final gini: 0.3063.
- Final distress: 0.4662.
- Final w_mb_eff: 0.4562.
- Final defection: 89.
- Final Q_net: 0.0157.
- Final reward: 0.0127.
- Diagnostic: 500 ticks, no NaN, no Inf, all states bounded.

## What changed from engine_v3

| Module | v3 (heuristic) | v4 (spec Part V.2) |
|---|---|---|
| Task | One-shot engage/avoid | Two-step with common/rare transition (P_COMMON=0.7) |
| Q_MB | `p_prop·max(Q_s2[:2]) + p_resp·max(Q_s2[2:])` | `Q_MB[a1] = Σ_s2 P_hat[a1, s2]·max_a2 Q_s2[s2, a2]` |
| P_hat | Not present | Per-agent (2, 2) transition tensor, EMA update with `eta_vol` |
| Q_MF update | `Q_mf[a1] += lr·(r − Q_mf[a1])` | `Q_mf[a1] += lr·(r + γ·max Q_mf − Q_mf[a1])` |
| γ source | None (γ = 1 implicit) | `γ = 0.5 + 0.45·tanh(k_DD/3)` |
| k_DD role | Ad-hoc learning-rate attenuator (wrong) | Correct: `k_DD` sets discount factor γ |

## Critical flags

### `[MF-GENERALIZED]`
The MF update now includes γ-bootstrap, matching spec Part V.2 but departing
from the hBayesDM ts_par7 implementation validated at G5 (which uses TD(0)
without γ). The G5 recovery result (ρ = 0.86) does NOT transfer directly to
this engine. A fresh recovery study on engine_v4's MF form is required before
w_MB values from engine_v4 can be cited alongside G5.

### `[COPULA-DEFECT-k_DD]`
The G2 copula's `k_DD` column has:
  min = −9.115, mean = −5.168, max = +3.178
This violates the spec's `k_DD > 0` constraint. Cause: an asymmetric
transform or mis-specified marginal in the G2 fit. Must be re-run before
any downstream artifact cites k_DD as empirically grounded. Until then,
engine_v4 uses a bounded tanh transform for γ.

### `[TANH-GAMMA-PROVISIONAL]`
The γ transform `0.5 + 0.45·tanh(k_DD/3)` is a bounded stopgap, not the
spec's `1/(1+k_DD)`. Once G2 is corrected, the engine must switch to the
canonical form and this flag is retired.

### `[SYNTH-PRIOR-COEFF]`
All coupling constants in engine_v4 (TAU_BASE, KAPPA_DECAY, C_BOOST,
E_ENGAGE, A_INHIB, T_THREAT, U_UNCERT, SM_DEFECT, SN_STATUS, SP_PUNISH,
SAL_GAIN, N_DISTRESS, LAMBDA_NORM_A, GAMMA_K) are provisional. Phase 6
must sweep each and report effect sizes.

## Scientific reading of the results

Cooperation dropped from 0.87 (v3) to 0.36 (v4) and Gini from 0.47 to 0.31.
These are consequences of the agent now having a real transition model
instead of a heuristic engagement bonus. Specifically:
- Q_MB is now genuinely sensitive to whether the agent's P_hat says the
  payoff state is reachable after the chosen stage-1 action.
- γ < 1 discounts future reward, so engagement is only attractive when the
  instantaneous expected reward is high.
- Rejections feed distress, which suppresses w_mb_eff, which further
  reduces MB engagement.

The result is a lower-cooperation, lower-Gini equilibrium than v3, but
stable, bounded, and reproducible.

## Out of scope (Phase 2d)

No unified trust + ultimatum (Phase 2e). No 2M scale test for the deep
agent (Phase 2f). No deterministic checkpoints (Phase 2g). No empirical
coefficient fitting (Phase 2h). No XAI trace API (Phase 3).
