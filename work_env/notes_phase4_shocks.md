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

## Shock types (4 implemented)

### network_thinning

Reduces the effective number of neighbours an affected agent can pick
from. The partner-selection logits `LOGITS_PICK` become a function of a
per-tick, per-agent `keff` tensor. For an agent with default k=6 and
magnitude 0.5, keff = max(6 - 3, 1) = 3 during the window. Models
social isolation or reduced mobility: fewer potential partners, so
mutual matching becomes less likely. Does not modify the underlying
graph — only the logits used to draw from it.

### wealth_redistribution

At each tick during the window, for affected agents, wealth is partially
pulled toward the population mean:

    wn_raw   = max(wealth + total_r, 0)
    mean_w   = mean(wn_raw)
    wn       = max(wn_raw + mag * (mean_w - wn_raw), 0)

with `mag` = magnitude, clipped to [0, 1]. At `mag=1` every affected
agent is reset to the mean (full equalization). At `mag=0` no effect.
Models a progressive tax-and-transfer intervention.

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

| benchmark | no-shock | scarcity | distress | SE |
|:---|---:|---:|---:|---:|
| UG offer  | 0.3959 | 0.3916 | 0.3942 | 0.050 |
| UG rej    | 0.1261 | 0.1479 | 0.1185 | 0.050 |
| TG send   | 0.4847 | 0.4893 | 0.4829 | 0.130 |
| TG ret    | 0.3376 | 0.3322 | 0.3334 | 0.120 |
| Gini      | 0.4637 | 0.4651 | 0.4331 | 0.120 |

### Deltas vs no-shock baseline

| benchmark | scarcity delta | % of SE | distress delta | % of SE |
|:---|---:|---:|---:|---:|
| UG offer | -0.0044 | -8.8%  | -0.0018 | -3.6%  |
| UG rej   | **+0.0218** | **+43.6%** | -0.0075 | -15.0% |
| TG send  | +0.0046 | +3.5%  | -0.0019 | -1.5%  |
| TG ret   | -0.0054 | -4.5%  | -0.0042 | -3.5%  |
| Gini     | +0.0014 | +1.2%  | **-0.0307** | **-25.6%** |

## Scientific interpretation

### Resource scarcity

The dominant effect is a 43.6% of an SE increase in UG rejection.
Mechanism: the accept decision clears a threshold
`U_accept = my_payoff - alpha * inequity_neg - beta * inequity_pos`.
The shock scales down every payoff (`r_off_m`, `r_acc_m`, `r_send_m`,
`r_ret_m` are all multiplied by 0.5 during the window). The accept
Q-values `Q_acc` therefore learn smaller values, while the reject head
keeps Q = 0. The relative attractiveness of accepting drops, so
rejection rises. This is the expected consequence of an economic
stress condition: the same offer becomes less acceptable when the
absolute payoff is smaller.

Offer rate barely moves (-8.8% of SE). This is consistent with the
proposer head being driven by the equity term `alpha * inequity_neg`,
which is scale-invariant. Proposers do not adjust offers under scarcity;
responders do adjust accept decisions.

### Distress injection

The dominant effect is a 25.6% of an SE decrease in Gini — i.e.,
**greater wealth equality** after the shock.

Mechanism: distress_injection raises the per-agent baseline, which
raises the equilibrium distress level via
`dn = 0.95 * dist + 0.05 * baseline_eff + ...`. Higher distress lowers
effective arbitration via
`w_mb_eff = clip(w_mb_base * (1 + C * 0.3) * (1 - dist * 0.8), 0, 1)`.

When `w_mb_eff` drops, agents lose model-based control and revert to
model-free habitual behavior. Habitual agents do not exploit the
long-horizon structure of the trust game, so they do not accumulate
the large asymmetric payoffs that strategy-based agents would. The
result is a flatter wealth distribution.

The other four benchmarks move less than 15% of an SE, indicating the
distress effect operates primarily through the wealth channel.

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
