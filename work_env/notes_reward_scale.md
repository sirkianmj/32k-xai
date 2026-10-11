# work_env — Reward-scale drift observation

**Date:** 11 October 2026.
**Status:** Documented, not fixed. Does not affect the current 5/5 PASS.
**Flag:** `[REWARD-SCALE-DRIFT]`.

## Observation

During the XAI trace work on the calibrated engine (`p6c_v5.py`), tracing
agent 100 at N=32000, T=500, seed=42 revealed monotone growth in wealth
and model-free Q values that is not visible in the aggregate benchmarks.

| tick | Q_mf_a0 | Q_net_a0 | wealth_pre |
| ---: | ---: | ---: | ---: |
| 100 | 0.233 | 0.222 | 36.7 |
| 300 | 5.903 | 5.636 | 685.7 |
| 500 | (see trace) | (see trace) | 1413.6 |

Starting wealth is `WEALTH_INIT = 10.0`. Over 500 ticks the agent's wealth
grows by a factor of ~140. Model-free Q for the engage action grows by
~2 orders of magnitude.

## Mechanism

The reward pipeline is:

    total_r = (r_off + r_acc + r_send + r_ret) * omega_social * (1 + sal + 0.05 * mood)
    wealth  = max(wealth + total_r, 0)

There is no cap, no decay, no redistribution. Positive rewards accumulate.
`REWARD_CLIP = 100` bounds the per-tick reward but not the cumulative
wealth. With `omega_social` around 2 for many agents, each successful
interaction contributes ~15-40 to wealth.

Separately, the arbitration weight `w_mb_eff` decays monotonically:

    w_mb_eff = clip(w_mb_base * (1 + C * 0.3) * (1 - distress * 0.8), 0, 1)

At `distress = 0.873` (which is exactly the equilibrium set by
`baseline_dist`), the modulator `(1 - distress*0.8) = 0.302`. For an
agent with `w_mb_base` near 0.2, `w_mb_eff` sits at ~0.05. Model-based
control is effectively off. Model-free dominates.

## Consequence for the softmax

Stage-1 action selection is:

    L1 = tau * [Q_net[:,0]+eng+per, Q_net[:,1]+avd]
    tau = TAU_BASE * (0.5 + O)

`TAU_BASE` is calibrated at `1.7889`. At tick 100 with `Q_net_a0 = 0.222`,
the logit is ~0.6, and the softmax provides meaningful exploration. At
tick 300 with `Q_net_a0 = 5.636`, the logit is ~15. The softmax is
effectively argmax.

The temperature `TAU_BASE` therefore operates over a range of logit
magnitudes that drifts by a factor of ~25 over the run. The calibration
targets the benchmark values measured at tick 500; those values are
matched. But the mechanism that produces them is not stationary.

## Why the 5/5 PASS is not affected

All five benchmarks are functions of aggregate behaviour at tick 500:

- UG mean offer, UG rejection, TG mean send, TG mean return, Final Gini.

Gini is scale-invariant in wealth. The other four are probabilities, not
magnitudes. None depends on the absolute scale of `Q_net`. So the
calibration remains valid for the benchmarks as stated.

## Why this matters for extrapolation

Any claim that depends on:

- the softmax providing constant exploration over the run,
- model-based and model-free arbitration operating at a stable weight,
- wealth or Q magnitudes being physically interpretable,

is not supported by the current engine. The `[REWARD-SCALE-DRIFT]` flag
should propagate to any downstream artifact that reasons about these.

## What is not proposed

Fixing this is not part of XAI-Trace-Integration. It would change the
calibration and require re-running P6c. If a fix is undertaken later, the
candidates are:

1. Normalize `Q_net` by a running estimate of its own scale before softmax.
2. Cap `wealth` or introduce redistribution.
3. Recalibrate `TAU_BASE` at a fixed reference magnitude.
4. Decay rewards with a half-life, converting cumulative to local.

Each has scientific consequences that require separate validation.
