# work_env — Phase 2c notes — 30-parameter behavioural agent (superseded)

**Commit:** `G-ENV-P2c` (retroactive — the code ran during 07 October 2026 but
the commit was deferred until after Phase 2d. Recorded here for provenance.)

**Status:** SUPERSEDED by `G-ENV-P2d`. Kept for audit trail.

## What engine_v3 was

First engine to read all 30 copula columns per tick. Heuristic MB module:
`Q_mb = p_prop·max(Q_s2[:2]) + p_resp·max(Q_s2[2:])`. This was NOT the spec's
Part V.2 Bellman backup. It was a role-weighted value estimate.

## Bugs found and fixed during P2c

1. **k_DD sign bug.** Original code: `lr_eff = lr / (1 + 0.1·k_DD)`. Copula
   `k_DD` ranges from −9.12 to +3.18. When `k_DD < 0`, `lr_eff > 1`, making
   TD non-contractive. Q_mf exploded to 1e13 within 50 ticks. Fixed via
   sigmoid + clip [0.01, 0.9].

2. **Mood drain bug.** Original: `reward = raw_reward·omega_soc·(1+sal·0.2) + mood·0.05`.
   The `mood·0.05` term applied every tick, including non-interacting ticks,
   draining agents with negative `m_t` (~-3.3) by 0.165/tick → −82 wealth over
   500 ticks. Fixed by folding mood into the interaction multiplier.

3. **Wealth unbounded below.** Wealth could go negative. Floored at 0.
   `[SYNTH-PRIOR-COEFF]` documented.

## Result at the time

32k × 500 ticks, wall 1.93 s, coop 0.8719, gini 0.4667, defection 26, all
states bounded, no NaN/Inf. State hash: `a3b15dd77f6323dfef635bf56c261fdbae86abef5f26204ec7f468e898bf9c0a`.

## Why superseded

The MB module was heuristic, not spec-compliant. `w_MB` was arbitrating
between `Q_mf` and a role-weighted estimate, not a true MDP value. This
violates spec Part V.2. `engine_v4` replaces it with a proper Bellman backup
over a learned `P_hat`. P2c result is retained for reference only — do not
cite the 0.87 coop figure as a scientific finding.

## Flags

`[SYNTH-PRIOR-COEFF]` on all coupling constants (14 items).
`[HEURISTIC-MB]` on the v3 MB module. Retired in v4.
`[SUPERSEDED-BY-P2D]` on the entire phase.
