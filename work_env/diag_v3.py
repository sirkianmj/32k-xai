# Diagnostic: run engine_v3 at N=1000 for 500 ticks, print state ranges every 50.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import numpy as np, jax, jax.numpy as jnp
from jax import random, lax
from pathlib import Path
from engine_v3 import load_all, build_ws, make_step, WS_N

N = 1000
T = 500
PARAMS = Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"

P_np = load_all(PARAMS)
# Take first N agents
P_small = {k: v[:N] for k, v in P_np.items()}
P = {k: jnp.asarray(v) for k, v in P_small.items()}

# Rebuild a small WS graph
import networkx as nx
G = nx.watts_strogatz_graph(n=N, k=6, p=0.1, seed=0)
degs = np.array([d for _, d in G.degree()])
K_max = int(degs.max())
nb = np.full((N, K_max), -1, dtype=np.int32)
for i, adj in G.adjacency():
    nbrs = sorted(adj.keys())
    nb[i, :len(nbrs)] = nbrs
neighbors = jnp.asarray(nb, jnp.int32)

# Print ranges of every static parameter
print("=== copula ranges (N=%d) ===" % N)
for k in sorted(P_small.keys()):
    v = P_small[k]
    print(f"  {k:18s} min={v.min():+.4f}  mean={v.mean():+.4f}  max={v.max():+.4f}")

Q_mf0   = jnp.zeros((N, 2), jnp.float32)
Q_s2_0  = jnp.zeros((N, 4), jnp.float32)
P_role0 = jnp.full((N, 2), 0.5, jnp.float32)
wealth0 = jnp.full((N,), 10.0, jnp.float32)
baseline = (P['general_distress'] + 0.3*P['DASS_Dep']
            + 0.3*P['DASS_Anx'] + 0.3*P['DASS_Str']) / 1.9
last0 = jnp.zeros((N,), jnp.int32)

step_fn = make_step(neighbors, P)
keys = random.split(random.PRNGKey(42), T)

@jax.jit
def run_inner(carry, ks):
    return lax.scan(step_fn, carry, ks)

print("\n=== running 500 ticks with per-50 diagnostics ===")
carry = (Q_mf0, Q_s2_0, P_role0, wealth0, baseline, last0)
for chunk_start in range(0, T, 50):
    chunk_keys = keys[chunk_start:chunk_start+50]
    carry, outs = run_inner(carry, chunk_keys)
    jax.block_until_ready(carry)
    Q_mf, Q_s2, P_role, wealth, distress, _ = carry
    nan_qmf = int(jnp.isnan(Q_mf).sum())
    nan_qs2 = int(jnp.isnan(Q_s2).sum())
    nan_d   = int(jnp.isnan(distress).sum())
    inf_qmf = int(jnp.isinf(Q_mf).sum())
    inf_qs2 = int(jnp.isinf(Q_s2).sum())
    inf_d   = int(jnp.isinf(distress).sum())
    print(f"tick {chunk_start+50:>4}  "
          f"Q_mf[min={float(Q_mf.min()):+.3f} max={float(Q_mf.max()):+.3f}] "
          f"Q_s2[min={float(Q_s2.min()):+.3f} max={float(Q_s2.max()):+.3f}] "
          f"W[min={float(wealth.min()):+.2f} max={float(wealth.max()):+.2f}] "
          f"D[min={float(distress.min()):+.3f} max={float(distress.max()):+.3f}] "
          f"NaNs(Q_mf={nan_qmf} Q_s2={nan_qs2} D={nan_d}) "
          f"Infs(Q_mf={inf_qmf} Q_s2={inf_qs2} D={inf_d})")
