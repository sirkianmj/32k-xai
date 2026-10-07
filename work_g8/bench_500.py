import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

import jax
import jax.numpy as jnp
from jax import random, lax, vmap
import time, pickle, traceback
from pathlib import Path

OUT = Path.home() / "projects/xai_project/work_g8"

P_COMMON = 0.7
T_STEPS  = 100                              # module-level → captured by closure
P_REWARD = jnp.array([[0.7, 0.3], [0.3, 0.7]])
AGENT_COUNTS = [100, 500, 1000, 2000]

def per_agent_sim(params, key):
    alpha, beta, phi, w_mb = params

    def step(carry, t_key):
        Q_mf, Q_td, last_a1 = carry
        k1, k2, k3, k4 = random.split(t_key, 4)

        m0 = jnp.max(Q_td[0, :])
        m1 = jnp.max(Q_td[1, :])
        Q_mb = jnp.stack([P_COMMON * m0 + (1.0 - P_COMMON) * m1,
                          P_COMMON * m1 + (1.0 - P_COMMON) * m0])

        Q_net = w_mb * Q_mb + (1.0 - w_mb) * Q_mf

        pers = jnp.where(last_a1 == 0, phi, 0.0) * jnp.array([1.0, 0.0]) + \
               jnp.where(last_a1 == 1, phi, 0.0) * jnp.array([0.0, 1.0])

        a1 = random.categorical(k1, beta * (Q_net + pers))
        s2 = jnp.where(random.uniform(k2) < P_COMMON, a1, 1 - a1)
        a2 = random.categorical(k3, beta * Q_td[s2, :])
        r  = random.bernoulli(k4, P_REWARD[s2, a2]).astype(jnp.float32)

        Q_td = Q_td.at[s2, a2].set(Q_td[s2, a2] + alpha * (r - Q_td[s2, a2]))
        Q_mf = Q_mf.at[a1].set(Q_mf[a1] + alpha * (r - Q_mf[a1]))

        return (Q_mf, Q_td, a1), (a1, a2, r)

    keys = random.split(key, T_STEPS)
    init = (jnp.zeros(2, jnp.float32),
            jnp.zeros((2, 2), jnp.float32),
            jnp.int32(-1))
    _, (A1, A2, R) = lax.scan(step, init, keys)
    return A1, A2, R

run_batch = jax.jit(vmap(per_agent_sim))

def sample_params(n_agents, seed):
    k1, k2, k3, k4 = random.split(random.PRNGKey(seed), 4)
    return jnp.stack([
        random.uniform(k1, (n_agents,), minval=0.05, maxval=0.9),
        random.uniform(k2, (n_agents,), minval=1.0,  maxval=8.0),
        random.uniform(k3, (n_agents,), minval=-0.5, maxval=0.5),
        random.uniform(k4, (n_agents,), minval=0.0,  maxval=1.0),
    ], axis=1)

print("=== G8: JAX vectorized two-step engine benchmark ===")
print(f"Device:  {jax.devices()[0]}")
print(f"Backend: {jax.default_backend()}")
print(f"T_STEPS: {T_STEPS}")
print()

# Warm-up
t0 = time.time()
_ = run_batch(sample_params(4, 0), random.split(random.PRNGKey(42), 4))[0].block_until_ready()
print(f"Warm-up compile: {time.time()-t0:.2f}s")
print()

results = []
for n in AGENT_COUNTS:
    params = sample_params(n, 1)
    keys   = random.split(random.PRNGKey(999), n)
    try:
        t0 = time.time()
        out = run_batch(params, keys)
        out[0].block_until_ready()
        wall = time.time() - t0

        t0 = time.time()
        out = run_batch(params, keys)
        out[0].block_until_ready()
        wall_cached = time.time() - t0

        A1, A2, R = out
        results.append({
            "n_agents": n,
            "wall_first_s": wall,
            "wall_cached_s": wall_cached,
            "A1_shape": tuple(A1.shape),
            "R_mean": float(jnp.mean(R)),
            "A1_frac_action0": float(jnp.mean(A1 == 0)),
            "error": None,
        })
        print(f"  N={n:5d}  first={wall:6.2f}s  cached={wall_cached:6.3f}s  "
              f"R_mean={float(jnp.mean(R)):.3f}  A1_shape={tuple(A1.shape)}", flush=True)
    except Exception as e:
        print(f"  N={n:5d}  FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
        results.append({"n_agents": n, "wall_first_s": float("nan"),
                        "wall_cached_s": float("nan"), "A1_shape": None,
                        "R_mean": float("nan"), "A1_frac_action0": float("nan"),
                        "error": f"{type(e).__name__}: {str(e)[:200]}"})

budget_s = 60.0
gate = next((r for r in results if r["n_agents"] == 500), None)
if gate and gate["error"] is None:
    verdict = "PASS" if gate["wall_first_s"] < budget_s else "FAIL"
else:
    verdict = "FAIL (N=500 errored)"

print()
print("=== G8 verdict ===")
if gate and gate["error"] is None:
    print(f"  N=500 x {T_STEPS} steps, first (compile+run) = {gate['wall_first_s']:.2f}s")
    print(f"  Budget: < {budget_s:.0f}s per 100 steps")
    print(f"  Result: {verdict}")
    print(f"  Cached execution = {gate['wall_cached_s']:.3f}s")
else:
    print(f"  N=500 error: {gate['error'] if gate else 'missing'}")
    print(f"  Result: {verdict}")

with open(OUT / "bench_500_results.pkl", "wb") as fh:
    pickle.dump(results, fh)
with open(OUT / "bench_500_results.txt", "w") as fh:
    fh.write(f"device={jax.devices()[0]}\nbackend={jax.default_backend()}\n")
    fh.write(f"T_STEPS={T_STEPS}\nN_agents\tfirst_s\tcached_s\tR_mean\tA1_frac_action0\terror\n")
    for r in results:
        fh.write(f"{r['n_agents']}\t{r['wall_first_s']:.4f}\t{r['wall_cached_s']:.4f}\t"
                 f"{r['R_mean']:.4f}\t{r['A1_frac_action0']:.4f}\t{r['error']}\n")
    fh.write(f"gate_verdict={verdict}\n")
print("\nSaved bench_500_results.pkl + .txt")
