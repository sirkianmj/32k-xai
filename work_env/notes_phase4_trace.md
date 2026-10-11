# work_env — Phase 4 step 4: scenario-aware XAI trace

**Date:** 11 October 2026
**Deliverable:** `xai_trace_scenarios.py` — trace a specific agent inside a
named scenario with the same 56-field schema as `xai_trace_v2`.

## What it does

Adds two functions:

- `trace_in_scenario(scenario_name, agent_i, N, T, seed)` — wraps
  `xai_trace_v2.trace_agent` with the shock spec of a named scenario
  from `scenarios.SCENARIOS`. Returns the same trace dict plus
  `scenario_name` and `scenario_description`.
- `format_scenario_tick(tr, tick_idx)` — prints the shock state at that
  tick (active shocks, resource modifier, baseline modifier, keff
  reduction, redistribution coefficient) followed by the standard trace
  dump.

The `trace_agent` function in `xai_trace_v2` gained a `shocks=None`
parameter. When `None`, the code path is byte-identical to the
pre-shock version. This is verified in two separate ways:

1. **Same copula, pre-shock code vs post-shock code.** Running the
   current patched code against the OLD copula (`synthetic_32k.pkl`)
   reproduces the canonical pre-shock hash
   `905da509bf037d50e2565b807abc6b838bc61d64c439841cede337f0f312eaea`.
   This confirms that the shock-tensor threading and the per-tick keff
   mask do not alter the shocks=None trajectory.

2. **Same copula, P6d baseline.** Running the current code against the
   P6d copula (`synthetic_32k_p6d.pkl`) with `shocks=None` reproduces
   the P6d final benchmark values
   `[0.39593542, 0.12607908, 0.48474643, 0.33758473, 0.46371707]`
   byte-for-byte.

The hash on the P6d copula differs from the pre-shock hash because the
copula itself changed (P6d grounded 5 priors; 4 were retired). Same code
path, different input, different trace values. This is expected.

When a scenario is passed, the shock tensors are precomputed by the
same code path the simulator uses, so the trace's `reward_total` and
`baseline_dist` fields reflect the shock state at each tick.

## What is visible in the trace

For `resource_scarcity`: the `reward_total` field is scaled down during
the window. The pre-multiplier components `r_off`, `r_acc`, `r_send`,
`r_ret` are **not** affected — the shock applies after their sum.

For `distress_injection`: the `baseline_dist` field is inflated during
the window. All downstream fields (distress_pre, w_mb_eff) reflect the
higher baseline.

For `network_thinning`: `partner_id` may differ because fewer neighbours
are available to pick from. The stage-1 dynamics are otherwise unchanged.

For `wealth_redistribution`: the trace does not include wealth
post-shock. The `wealth_pre` field shows wealth at the start of the
tick, before the shock's per-tick modification. To see the redistribution
effect, inspect the final wealth array, not the per-tick trace.

## What this enables

A researcher can now ask, for any scenario, any agent, any tick:

- What were the agent's Q-values at the decision?
- What was the arbitration weight?
- What did the partner do?
- Was the shock active at this tick? What modifier did it apply?

The answers come from the same 56-field trace that the pre-shock
platform produced. The schema is unchanged; the shock state is added as
context, not as additional fields.

## Example

    from xai_trace_scenarios import trace_in_scenario, format_scenario_tick
    tr = trace_in_scenario("resource_crisis", agent_i=100)
    print(format_scenario_tick(tr, 275))

Output shows the 5 benchmark values that the full simulation produced,
the scenario description, the active shocks at tick 275 (resource
scarcity, coefficient 0.4), and the 56-field per-tick decision record.

## Limitations

1. The trace still emits one agent at a time. To compare two agents
   under the same scenario, call `trace_in_scenario` twice.

2. The `wealth_redistribution` shock is not visible per-tick in the
   trace (see above). This is consistent with the `[WEALTH-INERT-IN-DECISION-LOOP]`
   finding: the shock changes wealth, but wealth does not feed back
   into any traced decision variable.

3. Determinism is preserved: same scenario + same agent + same seed →
   same trace bytes. The shock tensors are precomputed deterministically
   from the shock spec and `target_seed`; identical specs produce
   identical trace bytes.
