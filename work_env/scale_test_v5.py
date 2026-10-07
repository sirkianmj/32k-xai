# scale_test_v5.py — Phase 2f: deep agent scale-invariance at 32k -> 2M.
# Uses engine_v5.make_step directly with a per-N resampled copula and WS graph.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
import time, json, gc
from pathlib import Path
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
from jax import random, lax
from engine_v5 import make_step, load_all, build_ws, P_COMMON, N_ACTIONS, WEALTH_INIT

OUT = Path(__file__).parent
COPULA = Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"
N_TICKS = 500
SEED = 42
N_LADDER = [32000, 100000, 500000, 1000000, 2000000]

def _vram_mb():
    try:
        return jax.devices()[0].memory_stats().get("bytes_in_use", 0) / 1e6
    except Exception:
        return float("nan")

def run_one(N, src, seed=42):
    print(f"\n===== N = {N:,} =====", flush=True)
    jax.clear_caches(); gc.collect()
    rng = np.random.default_rng(seed + N)
    idx = rng.integers(0, len(src['O']), size=N)
    P = {k: jnp.asarray(v[idx]) for k, v in src.items()}
    t_g = time.time()
    nb, degs = build_ws(n=N, k=6, p=0.1, seed=0)
    print(f"  graph in {time.time()-t_g:.2f}s  deg[{degs.min()},{degs.max()}] mean={degs.mean():.3f}", flush=True)
    neighbors = jnp.asarray(nb, jnp.int32)
    Q_mf0  = jnp.zeros((N, 2), jnp.float32)
    Q_s2_0 = jnp.zeros((N, 2, N_ACTIONS), jnp.float32)
    P_hat0 = jnp.full((N, 2, 2), 0.0, jnp.float32)
    P_hat0 = P_hat0.at[:, 0, :].set(jnp.array([P_COMMON, 1.0 - P_COMMON]))
    P_hat0 = P_hat0.at[:, 1, :].set(jnp.array([1.0 - P_COMMON, P_COMMON]))
    wealth0 = jnp.full((N,), WEALTH_INIT, jnp.float32)
    baseline = (P['general_distress'] + 0.3*P['DASS_Dep']
                + 0.3*P['DASS_Anx'] + 0.3*P['DASS_Str']) / 1.9
    last0 = jnp.zeros((N,), jnp.int32)
    step_fn = make_step(neighbors, P)
    keys = random.split(random.PRNGKey(seed), N_TICKS)
    @jax.jit
    def run_inner(carry, ks): return lax.scan(step_fn, carry, ks)
    carry0 = (Q_mf0, Q_s2_0, P_hat0, wealth0, baseline, last0)
    t0 = time.time()
    try:
        final, outs = run_inner(carry0, keys)
        jax.block_until_ready(final)
    except Exception as e:
        print(f"  FAILED N={N}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return {"N": N, "error": f"{type(e).__name__}: {str(e)[:200]}", "vram_mb": _vram_mb()}
    wall = time.time() - t0
    print(f"  wall = {wall:.3f}s  VRAM = {_vram_mb():.1f} MB", flush=True)
    def _n(a): return np.asarray(a)
    return {"N": N, "wall_s": wall, "vram_mb": _vram_mb(),
            "coop_rate": _n(outs[0]), "gini": _n(outs[5]),
            "mean_distress": _n(outs[2]), "mean_w_mb_eff": _n(outs[3]),
            "defection": _n(outs[4]), "mean_reward": _n(outs[7]),
            "mean_a2": _n(outs[8])}

if __name__ == "__main__":
    print(f"Device: {jax.devices()[0]}  Backend: {jax.default_backend()}", flush=True)
    src = load_all(COPULA)
    print(f"copula: {len(src)} columns", flush=True)
    results = []
    for N in N_LADDER:
        r = run_one(N, src)
        results.append(r)
        with open(OUT / "scale_v5_results.json", "w") as f:
            json.dump([{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()} for r in results], f)
    print("\n\n=== SCALE LADDER (engine_v5, deep agent) ===")
    print(f"{'N':>10}  {'wall_s':>8}  {'vram_MB':>9}  {'coop':>7}  {'gini':>7}  {'dist':>7}  {'a2':>7}  {'reward':>7}")
    for r in results:
        if "error" in r:
            print(f"{r['N']:>10,}  FAILED: {r['error'][:60]}")
            continue
        print(f"{r['N']:>10,}  {r['wall_s']:>8.3f}  {r['vram_mb']:>9.1f}  "
              f"{r['coop_rate'][-1]:>7.4f}  {r['gini'][-1]:>7.4f}  "
              f"{r['mean_distress'][-1]:>7.4f}  {r['mean_a2'][-1]:>7.4f}  "
              f"{r['mean_reward'][-1]:>7.4f}")
    print("\n=== SCALE-INVARIANCE vs N=32000 ===")
    base = next((r for r in results if r.get("N") == 32000 and "error" not in r), None)
    if base:
        print(f"{'N':>10}  {'RMSE_coop':>11}  {'RMSE_gini':>11}  {'RMSE_a2':>11}")
        for r in results:
            if r.get("N") == 32000 or "error" in r: continue
            rc = np.sqrt(np.mean((r['coop_rate'] - base['coop_rate'])**2))
            rg = np.sqrt(np.mean((r['gini'] - base['gini'])**2))
            ra = np.sqrt(np.mean((r['mean_a2'] - base['mean_a2'])**2))
            print(f"{r['N']:>10,}  {rc:>11.5f}  {rg:>11.5f}  {ra:>11.5f}")
    print(f"\nSaved: {OUT / 'scale_v5_results.json'}")
