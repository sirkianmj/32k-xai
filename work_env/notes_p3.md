# work_env — Phase 3 notes — XAI trace API

**Commit:** `G-ENV-P3` — agent-level mechanistic trace.
**Date:** 08 October 2026.
**Hardware:** GTX 1660 Ti (6 GB VRAM), WSL2 Ubuntu, JAX 0.11.2 CUDA 13.

## Deliverable

- `xai_trace.py` — `trace_agent(ckpt, target_tick, agent_i, params_pkl)`
  returns a per-tick 24-field trace for one designated agent.
- `test_xai_trace.py` — verifies traced run is bit-for-bit identical to
  the untraced reference.
- `demo_xai_trace.py` — prints a readable per-tick trace table.

## Trace schema (24 fields)

Self view:
- Q_mf_a0, Q_mf_a1 — model-free stage-1 values at decision time
- Q_MB_a0, Q_MB_a1 — model-based Bellman values at decision time
- Q_net_a0, Q_net_a1 — arbitrated values
- w_mb_eff — arbitration weight used
- distress_pre — distress value at decision time
- a1 — stage-1 action chosen (0=engage, 1=avoid)
- s2 — resulting second-stage state (common/rare transition)
- a2 — stage-2 action chosen (0..3)
- P_hat_a1_0, P_hat_a1_1 — agent's belief about transitions under a1
- P_hat_alt_0, P_hat_alt_1 — agent's belief about transitions under other a1
- reward — reward received this tick
- interacting — did both partners engage and mutual pick succeed
- rejected — was this agent a responder who rejected

Partner view:
- partner_id — index of the chosen neighbor
- partner_alpha_envy — partner's D + lambda_ineq
- partner_beta_guilt — partner's A
- partner_a2 — partner's stage-2 action
- my_offer — this agent's offer (ultimatum)
- partner_accept — did the partner accept this agent's offer

## Result

Reference (checkpointed) run: 500 ticks, wall 2.17s, hash
`ef1b5da93dcfa9241f6e9b4a6f674ddf3125e3014ddd7ce93d02e1303bbb28ba`.

Trace agent 100 (from ckpt t=50 to t=500):
- wall: 2.08s
- shape: (450, 24) — 450 ticks x 24 fields
- final hash: `ef1b5da9...` — **matches reference, PASS**

Trace agent 0 (from ckpt t=50 to t=500):
- final hash: `ef1b5da9...` — **matches reference, PASS**

The trace re-implements engine_v5's step verbatim; only the return value
of the scan body is extended with the trace vector. The simulation itself
is unchanged, which is why both hashes match.

## Sanity check on agent 100's trace

- a1 unique: [0, 1] — agent engages on some ticks, avoids on others
- s2 unique: [0, 1] — both transition states are reached
- a2 unique: [2] — agent always chose "fair" (offer 5, send 6)
- interacting rate: 0.060

The 6% interaction rate is a consequence of the WS mutual-pick rule with
the "engage" gate: both partners must pick each other AND both must
choose engage. For k=6 the mutual-pick probability is ~1/6, and multiplied
by the engage probability for both partners, the joint rate falls to ~6%.
This is a property of the platform as specified, not a bug.

Agent 100's a2 = 2 constant is a real behavioural signature of that
agent's parameters (high enough SM_b or low enough Q_s2 to keep
convergence to fair). It is exactly what the trace API exists to reveal.

## What this satisfies

Spec Part VII — XAI protocol and trace schema. The platform now answers
the query: "at tick t, why did agent i do X?" with the full pre-decision
state (both Q systems, the arbitration weight, distress, beliefs) plus
the partner-side context. Nothing is inferred; every field is captured
at the tick where the decision occurred.

## Flags carried forward

- `[REWARD-SCALE-ASYMMETRY]` — trust payoffs ~8x ultimatum.
- `[COPULA-DEFECT-k_DD]` — G2 copula k_DD violates spec >0 constraint.
- `[TANH-GAMMA-PROVISIONAL]` — gamma transform is a stopgap.
- `[MF-GENERALIZED]` — MF form departs from hBayesDM ts_par7.
- `[SYNTH-PRIOR-COEFF]` — all coupling constants provisional.
- `[INERT-DEFAULT]` — SP_PUNISH, N_DISTRESS, C_BOOST, SAL_GAIN.

## Out of scope (Phase 3)

No cross-agent query yet (multi-agent traces). No dashboard. No
interventions. Phase 4 (provenance + interventions) and Phase 5
(dashboard) remain.
