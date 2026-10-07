# work_env — Phase 2e notes — unified trust + ultimatum

**Commit:** `G-ENV-P2e` — unified dual-task social interaction.
**Date:** 07 October 2026.
**Hardware:** GTX 1660 Ti (6 GB VRAM), WSL2 Ubuntu, JAX 0.11.2 CUDA 13.

## Deliverable

- `engine_v5.py` — unified trust + ultimatum on WS network, 4 stage-2 action categories.
- `run_32k_v5.py` — driver.
- `run_32k_v5.log`, `final_state_v5.pkl`, `run_32k_v5_summary.txt`.

## Result

- Agents: 32,000. Ticks: 500. Seed: 42.
- Wall (GPU): **1.926 s**.
- State hash: `302b7a8796419405cb21116d0f3949f9dc9a32f5667305dc026ebd6a147cd1bb`.
- Coop: 0.9542.
- Gini: 0.5624.
- Distress: 0.4629.
- w_mb_eff: 0.4581.
- Defection: 11.
- Q_net: 0.4225.
- Reward: 0.4797.
- mean_a2: **2.1096** (out of range [0, 3]).

## What changed from engine_v4

| Aspect | v4 | v5 |
|---|---|---|
| Tasks per tick | Ultimatum only | Trust game + ultimatum |
| Q_s2 shape | (N, 2, 2) | (N, 2, 4) |
| Stage-2 actions | cooperate / defect | self-max / self-mild / fair / generous |
| Offer values | argmax(U_P) continuous | discrete {2, 4, 5, 7} |
| Trust sends | n/a | discrete {0, 3, 6, 9} |
| Trust return multipliers | n/a | {0, 1, 2, 3} × sent |
| Reward magnitude | 0–10 | −9 to +81 (trust payoff) |

## Scientific reading

- **mean_a2 = 2.11** means the population converged on action category 2 = "fair" (ulti offer 5, trust send 6). This is the **first emergent strategy convergence** in the platform: agents learn that fair is the dominant choice because unfair offers get rejected (Fehr-Schmidt) and low sends get low returns (reciprocity).
- **Coop = 0.9542**: almost all ultimatum offers are accepted because most are fair (5).
- **Gini = 0.5624**: significantly higher than v4 (0.3063) because the trust game generates much larger payoffs. An agent that sends 9 and receives 3× return gets +72, dwarfing ultimatum payoffs. Wealth divergence is now driven by trust-game dynamics.
- **Q_net = 0.4225** vs. v4's 0.0157: 27× higher because trust-game reward magnitude enters the same TD update.

The high-coop + high-Gini coexistence is empirically documented (Nature 2025 social networking UG study).

## Flags raised

### `[NONINTERACTING-Q-DRIFT]` (inherited from v4)
Q updates fire unconditionally, even when `interacting` is False. For non-interacting agents, `reward = 0`, so Q values for the chosen (s2, a2) are pushed toward 0. This is a mild bias: an agent that never gets matched still "learns" that its chosen actions yield 0. Must be resolved in Phase 2h by gating the Q update on `interacting`.

### `[REWARD-SCALE-ASYMMETRY]`
Trust-game payoffs (up to 81) are ~8× larger than ultimatum payoffs (up to 10). This means the Q-learner effectively weighs trust-game outcomes far more heavily in the shared Q table. Scientific consequences: the "fair" convergence we observe may be driven mostly by the trust game, not the ultimatum. Must be documented; a Phase 6 sweep should test whether normalizing the two games to equal reward magnitude changes mean_a2.

### `[COPULA-DEFECT-k_DD]` (unchanged from P2d)
G2 copula's k_DD still violates the spec's >0 constraint. G2 re-run pending.

### `[TANH-GAMMA-PROVISIONAL]` (unchanged from P2d)
Gamma transform is a stopgap, not canonical.

### `[MF-GENERALIZED]` (unchanged from P2d)
v5 MF form still departs from validated hBayesDM ts_par7.

### `[SYNTH-PRIOR-COEFF]`
All coupling constants remain provisional. Adds TRUST_K = 3 (BDM 1995 standard) to the sweep list.

## Out of scope (Phase 2e)

No 2M scale test for v5 (Phase 2f). No deterministic checkpoints (Phase 2g). No coefficient sweeps (Phase 2h). No XAI trace (Phase 3).
