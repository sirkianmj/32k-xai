# work_env — Phase 2i notes — bug fixes + baseline re-establishment

**Commit:** `G-ENV-P2i` — SP_PUNISH live, Q-updates gated on interacting.
**Date:** 07 October 2026.
**Hardware:** GTX 1660 Ti (6 GB VRAM), WSL2 Ubuntu, JAX 0.11.2 CUDA 13.

## Deliverable

- `engine_v5.py` patched (backup at `engine_v5.py.pre_p2i`).
- `quick_sweep_sp.py` — SP_PUNISH-only re-sweep post-fix.
- `quick_sweep_sp.json` — sweep results.

## Fixes applied

### Fix 1 — [SP-PUNISH-DEAD-CODE] resolved

Before:
    dark_bias = -SM_b - SN_b + SP_b * 0.0

The trailing `* 0.0` zeroed SP_PUNISH entirely. After the fix, S_P
(psychopathy) reduces punishment sensitivity by scaling negative
Q_s2 values toward zero:

    q_s2_actions = jnp.where(
        q_s2_actions < 0.0,
        q_s2_actions * (1.0 - SP_b[:, None] * 0.5),
        q_s2_actions)

Broadcast corrected to SP_b[:, None] to match Q_s2's (N, 4) shape.

### Fix 2 — [NONINTERACTING-Q-DRIFT] resolved

Before: Q_mf, Q_s2, and P_hat updates fired unconditionally, including
on ticks where the agent was not matched or chose to avoid. Non-interacting
agents received reward=0, so their Q values were pulled toward zero,
creating a spurious "avoid is bad" signal.

After: all three updates are gated on `interacting`:

    update_mask = interacting.astype(jnp.float32)
    Q_s2_new_ = Q_s2_prev + update_mask * lr * (reward - Q_s2_prev)
    Q_mf_new_ = Q_mf_prev + update_mask * lr * (reward + gamma*v_next - Q_mf_prev)
    P_upd = P_at + update_mask[:, None] * eta_vol[:, None] * (one_hot - P_at)

This is the correct TD semantics: no state transition means no update.

## Baseline shift

| Metric | P2e (pre-fix) | P2i (post-fix) | Delta |
|---|---|---|---|
| state_hash | 302b7a87... | ef1b5da9... | (expected) |
| coop | 0.9542 | 0.9625 | +0.008 |
| gini | 0.5624 | 0.4394 | -0.123 |
| distress | 0.4629 | 0.4640 | ~0 |
| w_mb_eff | 0.4581 | 0.4576 | ~0 |
| defection | 11 | 33 | +22 |
| Q_net | 0.4225 | 15.7665 | +37x |
| reward | 0.4797 | 1.6479 | +3.4x |
| mean_a2 | 2.1096 | 2.0878 | -0.022 |

The Q_net and reward increases are direct consequences of fix 2:
Q values are no longer being dragged to zero on non-interaction ticks,
so they accumulate toward the true expected reward of cooperation.
The Gini drop from 0.56 to 0.44 follows: without the drift-induced
pessimism, non-interacting agents re-engage more readily and wealth
distributes more evenly.

## SP_PUNISH post-fix re-sweep

| mult | value | coop | gini | a2 | reward |
|---|---|---|---|---|---|
| x0.0 | 0.000 | 0.9634 | 0.4314 | 2.1148 | 1.6712 |
| x0.5 | 0.150 | 0.9634 | 0.4314 | 2.1153 | 1.6712 |
| x1.0 | 0.300 | 0.9634 | 0.4314 | 2.1153 | 1.6712 |
| x2.0 | 0.600 | 0.9634 | 0.4314 | 2.1157 | 1.6712 |

Sensitivity: dCoop=0.0000 dGini=0.0000 dA2=0.0005 score=0.0005
**Verdict: INERT** (confirmed, not artifact).

Reason: Q_s2 values remain predominantly positive because the cooperative
equilibrium produces positive rewards. S_P's "reduce punishment sensitivity"
branch only fires on negative Q values, which almost never occur. SP_PUNISH
is now confirmed [INERT-DEFAULT] rather than dead code.

## Determinism re-verification

`test_replay.py` re-run: 9/9 replay hashes match the new reference
`ef1b5da9...`. PASS. Bit-for-bit determinism holds after the fixes.

## Flags

### Resolved
- `[SP-PUNISH-DEAD-CODE]` — retired.
- `[NONINTERACTING-Q-DRIFT]` — retired.

### Upgraded
- `[INERT-DEFAULT]` for SP_PUNISH — confirmed inert, not artifact.

### Carried forward
- `[REWARD-SCALE-ASYMMETRY]`, `[COPULA-DEFECT-k_DD]`,
  `[TANH-GAMMA-PROVISIONAL]`, `[MF-GENERALIZED]`, `[SYNTH-PRIOR-COEFF]`.

## Environment status

All planned Phase 2 environment work is complete:
- 32k deep dual-system agents, unified trust + ultimatum.
- Scale-invariant to 2M.
- Deterministic replay from any checkpoint.
- All 30 copula parameters wired.
- Coefficient sensitivity ranked (P2h).
- Two downstream bugs fixed (P2i).

Next phase: **Phase 3 — XAI trace API**.
