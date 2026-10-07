# scale_test.py — Phase 2b.1 scale ladder
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import time, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random, lax
import networkx as nx
from engine_v2 import make_step, _find_col

OUT = Path(__file__).parent
COPULA_PKL = Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"
N_TICKS = 500
SEED = 42
N_LADDER = [32000, 100000, 500000, 1000000, 2000000]

def _vram_mb():
    try:
        s = jax.devices()[0].memory_stats()
        return s.get("bytes_in_use", 0) / 1e6
    except Exception:
        return float("nan")

def load_copula_source():
    df = pd.read_pickle(COPULA_PKL)
    a_col = _find_col(df.columns, 'alpha_R', 'alpha')
    b_col = _find_col(df.columns, 'beta_choice', 'beta')
    w_col = _find_col(df.columns, 'w_MB', 'w_mb_base', 'w_mb')
    d_col = _find_col(df.columns, 'general_distress', 'distress')
    for n, c in [('alpha', a_col), ('beta', b_col), ('w_mb', w_col), ('distress', d_col)]:
        if c is None: raise KeyError(f"missing {n}; cols={list(df.columns)}")
    return df[[a_col, b_col, w_col, d_col]].rename(columns={
        a_col: 'alpha_R', b_col: 'beta_choice', w_col: 'w_MB', d_col: 'general_distress'})

def resample_population(src, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(src), size=n)
    return src.iloc[idx].reset_index(drop=True)

def build_ws(n, k=6, p=0.1, seed=0):
    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    degs = np.array([d for _, d in G.degree()])
    K_max = int(degs.max())
    nb = np.full((n, K_max), -1, dtype=np.int32)
    for i, adj in G.adjacency():
        nbrs = sorted(adj.keys())
        nb[i, :len(nbrs)] = nbrs
    return nb, degs

def run_one(N, src, seed=42):
    print(f"\n===== N = {N:,} =====", flush=True)
    jax.clear_caches(); gc.collect()
    df_N = resample_population(src, N, seed=seed + N)
    alpha  = jnp.asarray(df_N['alpha_R'].values, np.float32)
    beta   = jnp.asarray(df_N['beta_choice'].values, np.float32)
    w_mb   = jnp.asarray(df_N['w_MB'].values, np.float32)
    dist0  = jnp.asarray(df_N['general_distress'].values, np.float32)

    t_graph = time.time()
    nb, degs = build_ws(N)
    print(f"  graph built in {time.time()-t_graph:.2f}s  degree min={degs.min()} max={degs.max()} mean={degs.mean():.3f}", flush=True)
    neighbors = jnp.asarray(nb, jnp.int32)

    Q_mf0 = jnp.zeros((N, 2), jnp.float32)
    Q_td0 = jnp.zeros((N, 2, 2), jnp.float32)
    wealth0 = jnp.full((N,), 10.0, jnp.float32)
    last0 = jnp.zeros((N,), jnp.int32)

    step_fn = make_step(neighbors, alpha, beta, w_mb)
    keys = random.split(random.PRNGKey(seed), N_TICKS)

    @jax.jit
    def run(carry, ks): return lax.scan(step_fn, carry, ks)

    carry0 = (Q_mf0, Q_td0, wealth0, dist0, last0)
    t0 = time.time()
    try:
        fc, outs = run(carry0, keys)
        jax.block_until_ready(fc)
    except Exception as e:
        print(f"  FAILED at N={N}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return {"N": N, "error": f"{type(e).__name__}: {str(e)[:200]}", "vram_mb": _vram_mb()}
    wall = time.time() - t0
    print(f"  wall = {wall:.3f}s  VRAM_use_MB = {_vram_mb():.1f}", flush=True)

    def _n(a): return np.asarray(a)
    return {"N": N, "wall_s": wall, "vram_mb": _vram_mb(),
            "coop_rate": _n(outs[0]), "gini": _n(outs[5]),
            "mean_distress": _n(outs[2]), "mean_w_mb_eff": _n(outs[3]),
            "defection": _n(outs[4]), "mean_reward": _n(outs[7])}

if __name__ == "__main__":
    print(f"Device: {jax.devices()[0]}  Backend: {jax.default_backend()}", flush=True)
    src = load_copula_source()
    print(f"copula: {src.shape}  mu(alpha)={src.alpha_R.mean():.4f}  mu(beta)={src.beta_choice.mean():.4f}  "
          f"mu(w_MB)={src.w_MB.mean():.4f}  mu(dist)={src.general_distress.mean():.4f}", flush=True)

    results = []
    for N in N_LADDER:
        r = run_one(N, src)
        results.append(r)
        with open(OUT / "scale_test_results.json", "w") as f:
            json.dump([{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()} for r in results], f)

    print("\n\n=== SCALE LADDER SUMMARY ===")
    print(f"{'N':>10}  {'wall_s':>8}  {'vram_MB':>9}  {'coop':>7}  {'gini':>7}  {'distress':>9}  {'defect':>7}  {'reward':>7}")
    for r in results:
        if "error" in r:
            print(f"{r['N']:>10,}  FAILED: {r['error'][:60]}")
            continue
        print(f"{r['N']:>10,}  {r['wall_s']:>8.3f}  {r['vram_mb']:>9.1f}  "
              f"{r['coop_rate'][-1]:>7.4f}  {r['gini'][-1]:>7.4f}  "
              f"{r['mean_distress'][-1]:>9.5f}  {r['defection'][-1]:>7.0f}  "
              f"{r['mean_reward'][-1]:>7.4f}")

    print("\n=== SCALE-INVARIANCE vs N=32000 (per-tick RMSE, normalised) ===")
    base = next((r for r in results if r.get("N") == 32000), None)
    if base is None:
        print("baseline missing.")
    else:
        print(f"{'N':>10}  {'RMSE_coop':>11}  {'RMSE_gini':>11}  {'RMSE_dist':>11}  {'max|dCoop|':>11}  {'max|dGini|':>11}")
        for r in results:
            if r.get("N") == 32000 or "error" in r: continue
            rc = np.sqrt(np.mean((r['coop_rate'] - base['coop_rate'])**2))
            rg = np.sqrt(np.mean((r['gini']      - base['gini'])**2))
            rd = np.sqrt(np.mean((r['mean_distress'] - base['mean_distress'])**2))
            mc = np.max(np.abs(r['coop_rate'] - base['coop_rate']))
            mg = np.max(np.abs(r['gini']      - base['gini']))
            print(f"{r['N']:>10,}  {rc:>11.5f}  {rg:>11.5f}  {rd:>11.5f}  {mc:>11.5f}  {mg:>11.5f}")

    print(f"\nSaved: {OUT / 'scale_test_results.json'}")
