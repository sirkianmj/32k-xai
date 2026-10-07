# engine_v5_sweep.py — Phase 2h: coupling-constant sensitivity sweep.
#
# For each axis in AXES, run engine_v5 at N_SMALL with the constant set to
# {base*0.5, base*1.0, base*2.0}, capture (coop, gini, mean_a2, wall).
# Then re-run the top-K load-bearing axes at N_FULL.
#
# "Load-bearing" = change in (coop, gini, mean_a2) from base->low/high
# exceeds a threshold. Inert constants can be left at base value.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
import time, json, gc
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from jax import random, lax
import engine_v5 as E

OUT = Path(__file__).parent
COPULA = Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"
N_SMALL = 10_000
N_FULL  = 32_000
N_TICKS = 500
SEED    = 42

AXES = [
    "TAU_BASE", "KAPPA_DECAY", "C_BOOST", "E_ENGAGE", "A_INHIB",
    "T_THREAT", "U_UNCERT", "SM_DEFECT", "SN_STATUS", "SP_PUNISH",
    "SAL_GAIN", "LAMBDA_NORM_A", "N_DISTRESS", "TRUST_K",
]
# GAMMA_K defined but not referenced in engine_v5 body. Dead. Flagged.

def run_once(N, seed, tag=""):
    """Run engine_v5 at N agents, 500 ticks, return macro final values."""
    jax.clear_caches()
    src = E.load_all(COPULA)
    idx = np.random.default_rng(seed + N).integers(0, len(src['O']), size=N)
    P = {k: jnp.asarray(v[idx]) for k, v in src.items()}
    nb, _degs = E.build_ws(n=N, k=6, p=0.1, seed=0)
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
    keys = random.split(random.PRNGKey(seed), N_TICKS)
    @jax.jit
    def run_inner(carry, ks):
        return lax.scan(step_fn, carry, ks)

    carry0 = (Q_mf0, Q_s2_0, P_hat0, wealth0, baseline, last0)
    t0 = time.time()
    final, outs = run_inner(carry0, keys)
    jax.block_until_ready(final)
    wall = time.time() - t0
    return {"coop": float(outs[0][-1]), "gini": float(outs[5][-1]),
            "a2": float(outs[8][-1]), "reward": float(outs[7][-1]),
            "dist": float(outs[2][-1]), "wall": wall}

def sweep_axis(axis, N, seed):
    base = float(getattr(E, axis))
    rows = []
    for mult in (0.5, 1.0, 2.0):
        val = base * mult
        saved = getattr(E, axis)
        try:
            setattr(E, axis, val)
            r = run_once(N, seed, tag=f"{axis}={val}")
        except Exception as ex:
            r = {"coop": float("nan"), "gini": float("nan"), "a2": float("nan"),
                 "reward": float("nan"), "dist": float("nan"), "wall": float("nan"),
                 "error": f"{type(ex).__name__}: {str(ex)[:120]}"}
        finally:
            setattr(E, axis, saved)
        rows.append({"axis": axis, "mult": mult, "value": val, **r})
        print(f"[sweep] {axis:15s} ×{mult:<4} = {val:<8.3f} "
              f"coop={r.get('coop',float('nan')):.4f} "
              f"gini={r.get('gini',float('nan')):.4f} "
              f"a2={r.get('a2',float('nan')):.4f} "
              f"wall={r.get('wall',float('nan')):.2f}s", flush=True)
    return rows

if __name__ == "__main__":
    print(f"Device: {jax.devices()[0]}  Backend: {jax.default_backend()}")
    print(f"Screen N={N_SMALL}  Full N={N_FULL}  Ticks={N_TICKS}")
    print()

    # ---- Phase 1: screen at N_SMALL ----
    print(f"=== SCREEN (N={N_SMALL}) ===")
    screen = []
    for axis in AXES:
        screen.extend(sweep_axis(axis, N_SMALL, SEED))
        print()

    # ---- Compute sensitivity per axis ----
    print("=== SENSITIVITY (max |Δ| from base) ===")
    sens = {}
    for axis in AXES:
        rows = [r for r in screen if r["axis"] == axis]
        base = next(r for r in rows if r["mult"] == 1.0)
        d_coop = max(abs(r["coop"] - base["coop"]) for r in rows if not np.isnan(r["coop"]))
        d_gini = max(abs(r["gini"] - base["gini"]) for r in rows if not np.isnan(r["gini"]))
        d_a2   = max(abs(r["a2"] - base["a2"])   for r in rows if not np.isnan(r["a2"]))
        score = d_coop + d_gini + d_a2
        sens[axis] = {"d_coop": d_coop, "d_gini": d_gini, "d_a2": d_a2, "score": score}
        print(f"  {axis:15s}  Δcoop={d_coop:.4f}  Δgini={d_gini:.4f}  Δa2={d_a2:.4f}  score={score:.4f}")

    # ---- Phase 2: full-resolution on top-3 load-bearing ----
    top = sorted(sens.items(), key=lambda kv: -kv[1]["score"])[:3]
    print(f"\n=== TOP-3 LOAD-BEARING: {[k for k,_ in top]} ===")
    full = []
    for axis, _ in top:
        full.extend(sweep_axis(axis, N_FULL, SEED))
        print()

    # ---- Save ----
    out = {"screen": screen, "full": full, "sensitivity": sens,
           "top_axes": [k for k, _ in top],
           "N_SMALL": N_SMALL, "N_FULL": N_FULL, "N_TICKS": N_TICKS, "SEED": SEED}
    with open(OUT / "sweep_p2h.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT / 'sweep_p2h.json'}")
