# Diagnostic: N=1000 for 500 ticks, print every 50, no NaN/Inf allowed.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
import numpy as np, jax, jax.numpy as jnp, networkx as nx
from jax import random, lax
from pathlib import Path
from engine_v4 import load_all, make_step, P_COMMON, WS_K, WS_P

N = 1000
T = 500
PARAMS = Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"
P_np = load_all(PARAMS)
P_small = {k: v[:N] for k, v in P_np.items()}
P = {k: jnp.asarray(v) for k, v in P_small.items()}

G = nx.watts_strogatz_graph(n=N, k=6, p=0.1, seed=0)
K_max = int(max(d for _, d in G.degree()))
nb = np.full((N, K_max), -1, dtype=np.int32)
for i, adj in G.adjacency():
    nb[i, :len(adj)] = sorted(adj.keys())
neighbors = jnp.asarray(nb, jnp.int32)

Q_mf0  = jnp.zeros((N, 2), jnp.float32)
Q_s2_0 = jnp.zeros((N, 2, 2), jnp.float32)
P_hat0 = jnp.full((N, 2, 2), 0.0, jnp.float32)
P_hat0 = P_hat0.at[:, 0, :].set(jnp.array([P_COMMON, 1.0 - P_COMMON]))
P_hat0 = P_hat0.at[:, 1, :].set(jnp.array([1.0 - P_COMMON, P_COMMON]))
wealth0 = jnp.full((N,), 10.0, jnp.float32)
baseline = (P['general_distress'] + 0.3*P['DASS_Dep']
            + 0.3*P['DASS_Anx'] + 0.3*P['DASS_Str']) / 1.9
last0 = jnp.zeros((N,), jnp.int32)

step_fn = make_step(neighbors, P)
keys = random.split(random.PRNGKey(42), T)

@jax.jit
def run_inner(carry, ks):
    return lax.scan(step_fn, carry, ks)

print("=== running 500 ticks with per-50 diagnostics ===")
carry = (Q_mf0, Q_s2_0, P_hat0, wealth0, baseline, last0)
for cs in range(0, T, 50):
    ck = keys[cs:cs+50]
    carry, outs = run_inner(carry, ck)
    jax.block_until_ready(carry)
    Q_mf, Q_s2, P_hat, wealth, distress, _ = carry
    def nf(x): return int(jnp.isnan(x).sum()), int(jnp.isinf(x).sum())
    n_qmf, i_qmf = nf(Q_mf)
    n_qs2, i_qs2 = nf(Q_s2)
    n_ph,  i_ph  = nf(P_hat)
    n_w,   i_w   = nf(wealth)
    n_d,   i_d   = nf(distress)
    print(f"tick {cs+50:>4}  "
          f"Q_mf[{float(Q_mf.min()):+.3f},{float(Q_mf.max()):+.3f}] "
          f"Q_s2[{float(Q_s2.min()):+.3f},{float(Q_s2.max()):+.3f}] "
          f"P_hat[{float(P_hat.min()):.3f},{float(P_hat.max()):.3f}] "
          f"W[{float(wealth.min()):.1f},{float(wealth.max()):.1f}] "
          f"D[{float(distress.min()):.3f},{float(distress.max()):.3f}]  "
          f"NaN=({n_qmf},{n_qs2},{n_ph},{n_w},{n_d})  "
          f"Inf=({i_qmf},{i_qs2},{i_ph},{i_w},{i_d})")
