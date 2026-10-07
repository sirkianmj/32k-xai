import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
import json
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from jax import random, lax
import engine_v5 as E

OUT = Path(__file__).parent
COPULA = Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"

def run_once(N, seed):
    jax.clear_caches()
    src = E.load_all(COPULA)
    idx = np.random.default_rng(seed + N).integers(0, len(src['O']), size=N)
    P = {k: jnp.asarray(v[idx]) for k, v in src.items()}
    nb, _ = E.build_ws(n=N, k=6, p=0.1, seed=0)
    neighbors = jnp.asarray(nb, jnp.int32)
    Q_mf0  = jnp.zeros((N, 2), jnp.float32)
    Q_s2_0 = jnp.zeros((N, 2, E.N_ACTIONS), jnp.float32)
    P_hat0 = jnp.full((N, 2, 2), 0.0, jnp.float32)
    P_hat0 = P_hat0.at[:, 0, :].set(jnp.array([E.P_COMMON, 1.0 - E.P_COMMON]))
    P_hat0 = P_hat0.at[:, 1, :].set(jnp.array([1.0 - E.P_COMMON, E.P_COMMON]))
    wealth0 = jnp.full((N,), E.WEALTH_INIT, jnp.float32)
    baseline = (P['general_distress'] + 0.3*P['DASS_Dep']
                + 0.3*P['DASS_Anx'] + 0.3*P['DASS_Str']) / 1.9
    last0 = jnp.zeros((N,), jnp.int32)
    step_fn = E.make_step(neighbors, P)
    keys = random.split(random.PRNGKey(seed), 500)
    @jax.jit
    def run_inner(carry, ks): return lax.scan(step_fn, carry, ks)
    carry0 = (Q_mf0, Q_s2_0, P_hat0, wealth0, baseline, last0)
    final, outs = run_inner(carry0, keys)
    jax.block_until_ready(final)
    return {"coop": float(outs[0][-1]), "gini": float(outs[5][-1]),
            "a2": float(outs[8][-1]), "reward": float(outs[7][-1])}

if __name__ == "__main__":
    base = float(getattr(E, "SP_PUNISH"))
    rows = []
    for mult in (0.0, 0.5, 1.0, 2.0):
        val = base * mult
        saved = getattr(E, "SP_PUNISH")
        try:
            setattr(E, "SP_PUNISH", val)
            r = run_once(10_000, 42)
        finally:
            setattr(E, "SP_PUNISH", saved)
        rows.append({"mult": mult, "value": val, **r})
        print(f"[sweep] SP_PUNISH x{mult:<4} = {val:<6.3f}  "
              f"coop={r['coop']:.4f}  gini={r['gini']:.4f}  "
              f"a2={r['a2']:.4f}  reward={r['reward']:.4f}", flush=True)
    with open(OUT / "quick_sweep_sp.json", "w") as f:
        json.dump({"axis": "SP_PUNISH", "base": base, "rows": rows}, f, indent=2)
    base_row = next(r for r in rows if r["mult"] == 1.0)
    d_coop = max(abs(r["coop"] - base_row["coop"]) for r in rows)
    d_gini = max(abs(r["gini"] - base_row["gini"]) for r in rows)
    d_a2   = max(abs(r["a2"]   - base_row["a2"])   for r in rows)
    score  = d_coop + d_gini + d_a2
    print(f"\nSP_PUNISH post-fix sensitivity:")
    print(f"  dCoop={d_coop:.4f}  dGini={d_gini:.4f}  dA2={d_a2:.4f}  score={score:.4f}")
    print(f"  Verdict: {'LOAD-BEARING' if score > 0.05 else 'INERT'}")
