# G8 — JAX vectorized two-step engine benchmark — notes

**Deliverable:** A vectorized JAX implementation of the two-step decision
architecture, timed on the GTX 1660 Ti against the G8 budget
(< 60 s per 500-agent × 100-step trace).

## G8 verdict

**PASS — with ~58× headroom.**

| N agents | first call (compile + run) | cached execution |
| ---: | ---: | ---: |
| 100  | 1.01 s | 0.003 s |
| 500  | 1.03 s | 0.004 s |
| 1000 | 1.00 s | 0.004 s |
| 2000 | 1.01 s | 0.013 s |

N=500 × 100 steps: **1.03 s** vs. 60 s budget. **58× under budget.**
Cached (post-compile) execution: **4 ms** per 500-agent × 100-step trace.

## Architecture

- **Per-agent function** (`per_agent_sim`): implements the two-step hybrid
  model from Daw et al. (2011). One agent, T timesteps.
  - **Model-free branch** (`Q_mf`): TD(0) update on stage-1 action values.
  - **Model-based branch** (`Q_td`): a 2×2 table of stage-2 state-action
    values; stage-1 model-based values are computed on the fly by
    marginalizing over the transition kernel (P_COMMON = 0.7).
  - **Arbitration:** `Q_net = w_mb * Q_mb + (1 - w_mb) * Q_mf`.
  - **Perseveration:** additive bonus `phi` on the last-executed stage-1
    action.
  - **Choice:** categorical (softmax) with inverse temperature `beta`.
  - **Reward:** Bernoulli with p = 0.7 for the diagonal, 0.3 off-diagonal.

- **Vectorization:** `jax.vmap(per_agent_sim)` maps over the agent
  dimension. No Python loops over agents.

- **Temporal loop:** `jax.lax.scan` over the T-step dimension. No Python
  loops over time.

- **JIT:** `run_batch = jax.jit(vmap(per_agent_sim))`. First call for a
  given (N, T_STEPS) triggers XLA compilation; subsequent calls hit the
  compiled executable.

## Memory hygiene

Initial run OOM'd at 4.5 GB (default XLA preallocation on a 6 GB card).
Fixed by setting, before importing JAX:

    XLA_PYTHON_CLIENT_PREALLOCATE=false
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

This allows JAX to allocate on demand up to ~5.1 GB instead of grabbing
most of VRAM at import.

## Design choices

1. **Module-level `T_STEPS` constant.** Earlier attempt passed T as a
   function argument, which made it a tracer inside the jitted function;
   `random.split(key, T)` requires a static integer. T_STEPS is now
   closed over at module scope.

2. **`random.split(key, T)` pre-generated.** Rather than splitting the key
   inside the scan step, we split once before the scan and consume one
   subkey per step. Cleaner XLA graph, deterministic.

3. **No python-level branching inside the loop.** Every conditional is
   `jnp.where`. This is what lets XLA fuse the whole temporal loop into
   a single kernel.

## Sanity checks

- **`R_mean ≈ 0.60`.** Reward matrix diagonal is 0.7. With agents learning
  to prefer diagonal (state 0 / action 0, state 1 / action 1), a mean
  reward above 0.5 is expected. Observed 0.597–0.605 → learning converges.

- **`A1_frac_action0 ≈ 0.5`.** Parameters (alpha, beta, phi, w_mb) are
  sampled uniformly and symmetrically; Q-tables initialize at zero. No
  systematic bias toward either stage-1 action is expected. Observed 0.50
  → choice rule is unbiased.

- **Shapes verified.** A1, A2, R all have shape `(N, 100)` for each N
  tested.

## Scaling extrapolation to 32k

The cached execution time scales roughly linearly with N: 4 ms at N=500,
13 ms at N=2000 (3.3× the compute for 4× the agents — sublinear at low N
due to kernel-launch overhead). Extrapolating to 32,000 agents at the
same T_STEPS=100: **~0.20 s** per trace. A full 1000-step trace at 32k
would be on the order of **~2 s** of pure compute.

The engine is not the bottleneck for scale-up. The bottleneck is the
Vine Copula parameter synthesis (Python/NumPy, CPU) and, above that, the
data I/O.

## Parameter space represented

The benchmark draws parameters from the empirically-informed ranges:

| Parameter | Range used | Source |
| :--- | :--- | :--- |
| alpha (learning rate) | [0.05, 0.90] | work_n500 real-data fit (0.37 ± 0.27) |
| beta (inverse temp) | [1.0, 8.0] | work_n500 real-data fit (4.65 ± 3.58) |
| phi (perseveration) | [-0.5, 0.5] | work_n500 real-data fit (0.30 ± 0.29) |
| w_mb (arbitration) | [0.0, 1.0] | work_n500 real-data fit (0.62 ± 0.33) |

The benchmark's purpose is a wall-clock feasibility check, not a scientific
statement. The parameter ranges are plausible but the samples are uniform,
not drawn from the fitted marginal distributions. The Vine Copula stage
(G2) will use the actual marginals.

## Caveats

1. **`beta` is the collapsed version.** In the fitted model, `alpha` and
   `beta` are shared across the two stages (`[COLLAPSED-LR]`,
   `[COLLAPSED-BETA]`). The engine here reflects that simplification.
   A future asymmetric version would double the parameter count but does
   not affect the compute benchmark.

2. **The model-free Q-table is 2-dimensional** (one entry per stage-1
   action, not per state-action pair). This matches the original Daw
   parameterization: model-free learning at stage 1 uses only the action,
   ignoring the state, which is the standard model-free specification for
   the two-step task.

3. **No belief-state tracking, no higher-order inference.** This is the
   minimal engine that satisfies G8. The full architecture has more
   machinery (state-space dual-system with DASS-modulated w_MB decay, XAI
   logging, etc.) that will need its own benchmarks.

## Reproducibility

Script: `bench_500.py`. Log: `bench_500.log`. Outputs:
`bench_500_results.pkl`, `bench_500_results.txt`. Runtime: ~5 s wall.
Environment variables set at script top: `XLA_PYTHON_CLIENT_PREALLOCATE`
and `XLA_PYTHON_CLIENT_MEM_FRACTION`.

    cd work_g8 && python bench_500.py

Hardware: NVIDIA GTX 1660 Ti, 6 GB VRAM, CUDA compute capability 7.5;
Python 3.14; JAX 0.11.2 (CUDA 13 build); driver 615.78.05.
