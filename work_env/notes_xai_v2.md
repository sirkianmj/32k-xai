# work_env — XAI Trace v2 (calibrated engine) — notes

**Phase:** XAI-Trace-Integration (Phase 1 of the post-P6c plan).
**Date:** 11 October 2026.
**Hardware:** GTX 1660 Ti (6 GB VRAM), WSL2 Ubuntu, JAX 0.11.2 CUDA 13.
**Baseline:** `p6c_v5.py` at commit `1cc5e19` (P6d rollback).

## Deliverables

- `p6c_v5.py` — patched with optional `trace_agent_i=None` argument on
  `build_simulator`. When `None`, behaviour is byte-identical to the
  version committed at `1cc5e19`. When an int, `_run_with_seed` returns
  `(metrics, traces)` instead of `metrics`.
- `xai_trace_v2.py` — driver that wraps the patched simulator and
  returns a per-tick dict for one designated agent.
- `_trace_hash_helper.py` — subprocess helper for the cross-process
  determinism check.
- `trace_agent100_seed42_v3.npy` — canonical archived trace.
- `xai_v2_patch2.log`, `xai_v2_v3fix2.log`, `xai_v2_determinism.log` —
  verification evidence.

## Trace schema (56 fields)

Grouped by role. Order in `p6c_v5.step` is authoritative; the schema
list in `xai_trace_v2.TRACE_V2_KEYS` matches that order field-for-field.

| Group | Count | Fields |
| :--- | ---: | :--- |
| Stage-1 self | 19 | Q_mf_a0, Q_mf_a1, Q_MB_a0, Q_MB_a1, Q_net_a0, Q_net_a1, w_mb_eff, tau, engage_adv, avoid_adv, persev, L1_a0, L1_a1, a1, s2, P_hat_a1_0, P_hat_a1_1, P_hat_alt_0, P_hat_alt_1 |
| Offer head | 3 | a_off, my_off, Q_off_chosen |
| Send head | 3 | a_send, my_send, Q_send_chosen |
| Accept head | 6 | a_acc, Q_acc_accept, Q_acc_reject, U_accept, inequity_neg, inequity_pos |
| Return head | 3 | a_ret, my_rm, Q_ret_chosen |
| Reward decomposition | 5 | r_off, r_acc, r_send, r_ret, reward_total |
| Partner view | 11 | partner_id, partner_alpha_e, partner_beta_g, partner_a_off, partner_a_send, partner_a_ret, rec_off, rec_send, rec_rm, partner_a_acc, partner_a1 |
| Events | 3 | interacting, rejected, is_mutual |
| Context | 3 | distress_pre, baseline_dist, wealth_pre |

## Non-interference

The trace patch does not perturb the calibrated simulation.

    untraced = [0.39458367 0.12024558 0.48318192 0.33410013 0.44285348]
    traced   = [0.39458367 0.12024558 0.48318192 0.33410013 0.44285348]
    max|d|   = 0.000e+00
    byte-identical: True

`p6c_verify2.py` after the patch re-produces `p6c_final.json` byte-for-byte.

## Determinism

Canonical trace: agent 100, N=32000, T=500, seed=42.

    sha256 = 905da509bf037d50e2565b807abc6b838bc61d64c439841cede337f0f312eaea

- Same-process (two calls in one process): byte-identical.
- Cross-process (two fresh subprocesses): byte-identical.
- Cross-process vs in-process: byte-identical.
- Agent 100 vs agent 200 (same seed): different hashes (expected).
- Agent 100 seed 42 vs seed 43: different hashes (expected).

## Semantic range checks

27 predicate checks on the trace values. All PASS.

- Integer-typed fields (`a1`, `s2`, `a_off`, `a_send`, `a_acc`, `a_ret`,
  `partner_a1`, `partner_a_acc`, `partner_a_off`, `partner_a_send`,
  `partner_a_ret`, `interacting`, `rejected`, `is_mutual`) lie in their
  declared discrete sets.
- Range-typed fields (`my_off`, `my_send`, `my_rm`, `rec_off`,
  `rec_send`, `rec_rm`, `w_mb_eff`, `inequity_neg`, `inequity_pos`)
  lie within their declared intervals.
- All Q-value fields are finite.

## What this satisfies

Spec Part VII (XAI protocol and trace schema). The platform can now
answer, for the calibrated engine: "at tick t, why did agent i do X?"
with the full pre-decision state — both Q systems, the arbitration
weight, the distress modulator, the four head Q-values, the accept
utility decomposition, the reward breakdown, and the partner-side
context — captured at the tick where the decision occurred.

## Limitations (documented, not defects)

1. **Internal RNG noise is not traced.** `random.categorical` draws are
   captured as the resulting discrete action, not as the Gumbel sample
   that produced them. This is by design: the sampled action is what
   the agent did, and the logits are traceable. The intermediate noise
   is not a decision variable.

2. **Cross-agent traces are not supported.** The API traces one agent
   at a time. `partner_*` fields describe that agent's partner at the
   current tick, but the partner's full decision vector is not returned.
   Multi-agent joint tracing is deferred to a future phase.

3. **Only the final action each tick is captured.** The four stage-2
   heads (offer, send, accept, return) all sample within one tick. Their
   chosen actions are traced. The order of sampling and the RNG split
   keys are visible in `p6c_v5.py` but not exposed as trace fields.

4. **The trace block adds ~50 lines to `p6c_v5.py`.** The patched file
   is 16.5 KB. The unpatched version is preserved at `p6c_v5.py.pre_trace3`.
   A single `cp` restores the pre-trace state.

## Flags raised

- `[XAI-TRACE-CALIBRATED-ENGINE]` — the trace now runs on `p6c_v5`.
- `[XAI-TRACE-56-FIELDS]` — schema v3.
- `[XAI-TRACE-DETERMINISTIC]` — canonical hash `905da509...`.

## Flags carried forward

- `[REWARD-SCALE-DRIFT]` — see `notes_reward_scale.md`.
- `[SYNTHETIC-PRIOR-PARAMS]` — 10 parameters still ungrounded.
- `[SIMPLIFIED MODEL]` — Fehr-Schmidt α recovery below threshold (G6).
- `[KOOL-VALIDATION]` — now PASSED, retires `[KOOL-VALIDATION PENDING]`.
- `[P6D-RANK-TRANSFORM-REJECTED]` — from the rollback commit.
