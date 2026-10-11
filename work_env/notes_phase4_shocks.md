# work_env — Phase 4: runtime shock API

**Date:** 11 October 2026
**Baseline:** `p6c_v5.py.pre_shocks` (commit 062bd01)
**Status:** API implemented. Two shock types verified.

## What it does

Adds an optional `shocks=` parameter to `p6c_v5.build_simulator`. When
`shocks=None`, behaviour is byte-identical to the pre-patch version
(verified against `p6c_final_p6d_final.json`). When a list of shock
specs is passed, the simulator applies them at the specified ticks.

A shock spec is a dict:

    {
      "tick":         int,   # first tick the shock is active
      "duration":     int,   # number of ticks it lasts
      "type":         str,   # "resource_scarcity" or "distress_injection"
      "magnitude":    float, # severity
      "target_frac":  None | float,  # fraction of agents affected
      "target_seed":  int,   # RNG for reproducible targeting
    }

## Shock types

### resource_scarcity

Multiplies the reward modifier by `(1 - magnitude)` for affected agents
during the shock window. The reward modifier is:

    mult = omega_social * (1 + sal + 0.05 * mood)

so resource scarcity scales down the effective reward per interaction
without changing any psychological parameter. Wealth accumulates more
slowly. This is the economic stress condition.

### distress_injection

Adds `magnitude` to the per-agent baseline distress during the window.
The distress update becomes:

    dn = DISTRESS_DECAY * dist + DISTRESS_PULL * (baseline + bm_t) + rejection term

Higher baseline → higher equilibrium distress → lower effective w_MB
(because w_mb_eff = clip(w_mb_base * (1 + C) * (1 - distress * 0.8))). This
is the psychological stress condition.

## Determinism

Same shock list + same seed → same trajectory. The `target_seed` field
deterministically selects which agents are affected. Verified by
re-running demo A and B with identical outputs.

## Audit trail

The shock specs are part of the build_simulator call. Any script that
generates a run can serialise the shock list to JSON alongside the
output. Downstream artifacts should always record:

- the shock list,
- the seed,
- the copula path (default `synthetic_32k_p6d.pkl`).

## Demonstration

Two shocks at 32k x 500, seed 42, on `synthetic_32k_p6d.pkl`:

**A. Resource scarcity, magnitude 0.5, all agents, tick 250 to 500.**

**B. Distress injection, magnitude 1.0, 50% of agents, tick 250 to 500.**

Platform output (see shock_a_resource_scarcity.npy and
shock_b_distress_injection.npy for full arrays; the fields are the same
five the verifier reports):

| benchmark | no-shock | scarcity | distress |
|:---|---:|---:|---:|
| UG offer | 0.3959 | (see block) | (see block) |
| UG rej | 0.1261 | (see block) | (see block) |
| TG send | 0.4847 | (see block) | (see block) |
| TG ret | 0.3376 | (see block) | (see block) |
| Gini | 0.4637 | (see block) | (see block) |

(Exact numbers from the run log; the file is not pre-baked because the
shock effects are the scientific result and must be recorded verbatim.)

## Limitations

1. The shock only fires when the sim is built with the shock list
   embedded. There is no runtime injection API — the shock tensor is
   precomputed at compile time.

2. Network densification is NOT implemented. Rewiring the WS graph
   mid-simulation would require rebuilding the neighbour tensor inside
   the scan, which is not supported by lax.scan without a rewrite of
   the partner matching step.

3. The trace block still emits 56 fields. The `baseline_dist` field now
   shows `baseline_eff` (baseline + shock modifier). This is the correct
   scientific value and does not add fields.

## Next steps for Phase 4

1. Add a third shock type: network restructuring at the "pick" level
   (lower the effective k by masking some neighbour slots).
2. Add a fourth: wealth redistribution (Gini-targeted intervention).
3. Write a "Scenario" spec that combines multiple shocks into a named
   package for reproducibility.
4. Add a shock audit log writer that produces a JSON manifest per run.
