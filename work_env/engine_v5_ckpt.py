# engine_v5_ckpt.py — Phase 2g: deterministic checkpoints + replay.
#
# Wraps engine_v5 with per-chunk checkpointing. Every `ckpt_every` ticks, the
# full carry (Q_mf, Q_s2, P_hat, wealth, distress, last_a1) is written to disk
# together with the base seed and the tick number. Replay from any checkpoint
# reproduces the exact same state as the original run.
#
# Design contract:
#   - run_with_checkpoints() runs the full sim in chunks, saving ckpts.
#   - replay_from_checkpoint() loads a ckpt and runs forward to any later tick.
#   - state_hash() over (Q_mf, Q_s2, P_hat, wealth, distress) must match
#     bit-for-bit between the original run and the replayed run.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import time, hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax

from engine_v5 import (
    load_all, build_ws, make_step,
    P_COMMON, N_ACTIONS, WEALTH_INIT, WS_N,
)

CKPT_KEYS = ["Q_mf", "Q_s2", "P_hat", "wealth", "distress", "last_a1"]

def state_hash(d):
    h = hashlib.sha256()
    for k in ["Q_mf", "Q_s2", "P_hat", "wealth", "distress"]:
        h.update(np.ascontiguousarray(d[k], np.float32).tobytes())
    return h.hexdigest()

def _carry_to_dict(carry):
    Q_mf, Q_s2, P_hat, wealth, distress, last_a1 = carry
    return {
        "Q_mf":    np.asarray(Q_mf),
        "Q_s2":    np.asarray(Q_s2),
        "P_hat":   np.asarray(P_hat),
        "wealth":  np.asarray(wealth),
        "distress":np.asarray(distress),
        "last_a1": np.asarray(last_a1),
    }

def _dict_to_carry(d, N):
    return (
        jnp.asarray(d["Q_mf"]),
        jnp.asarray(d["Q_s2"]),
        jnp.asarray(d["P_hat"]),
        jnp.asarray(d["wealth"]),
        jnp.asarray(d["distress"]),
        jnp.asarray(d["last_a1"]),
    )

def _init_carry(P, N):
    Q_mf0  = jnp.zeros((N, 2), jnp.float32)
    Q_s2_0 = jnp.zeros((N, 2, N_ACTIONS), jnp.float32)
    P_hat0 = jnp.full((N, 2, 2), 0.0, jnp.float32)
    P_hat0 = P_hat0.at[:, 0, :].set(jnp.array([P_COMMON, 1.0 - P_COMMON]))
    P_hat0 = P_hat0.at[:, 1, :].set(jnp.array([1.0 - P_COMMON, P_COMMON]))
    wealth0 = jnp.full((N,), WEALTH_INIT, jnp.float32)
    baseline = (P['general_distress'] + 0.3*P['DASS_Dep']
                + 0.3*P['DASS_Anx'] + 0.3*P['DASS_Str']) / 1.9
    last0 = jnp.zeros((N,), jnp.int32)
    return (Q_mf0, Q_s2_0, P_hat0, wealth0, baseline, last0)

def _prepare(P_np, N):
    """Slice copula to N agents, build WS graph at N, return (P, neighbors)."""
    P = {k: jnp.asarray(v[:N]) for k, v in P_np.items()}
    nb, degs = build_ws(n=N, k=6, p=0.1, seed=0)
    neighbors = jnp.asarray(nb, jnp.int32)
    return P, neighbors, degs

def run_with_checkpoints(params_pkl, n_ticks, ckpt_dir, ckpt_every=50, seed=42, N=WS_N):
    """Run full sim in chunks, saving checkpoint after each chunk."""
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    P_np = load_all(params_pkl)
    P, neighbors, degs = _prepare(P_np, N)
    print(f"[ckpt] N={N} degree[{degs.min()},{degs.max()}] mean={degs.mean():.3f}")
    step_fn = make_step(neighbors, P)
    keys = random.split(random.PRNGKey(seed), n_ticks)
    @jax.jit
    def run_inner(carry, ks):
        return lax.scan(step_fn, carry, ks)

    carry = _init_carry(P, N)
    saved = []
    t0 = time.time()
    for cs in range(0, n_ticks, ckpt_every):
        ce = min(cs + ckpt_every, n_ticks)
        chunk_keys = keys[cs:ce]
        carry, _outs = run_inner(carry, chunk_keys)
        jax.block_until_ready(carry)
        tick = ce
        d = _carry_to_dict(carry)
        h = state_hash(d)
        fname = ckpt_dir / f"tick_{tick:04d}.npz"
        np.savez(fname, **d, tick=tick, seed=seed, n_ticks=n_ticks, N=N, hash=h)
        saved.append({"tick": tick, "path": str(fname), "hash": h})
        print(f"[ckpt] saved t={tick:>4}  hash={h[:16]}...")
    wall = time.time() - t0
    print(f"[ckpt] wall = {wall:.3f}s  saved {len(saved)} checkpoints")
    return saved, wall

def replay_from_checkpoint(ckpt_path, params_pkl, target_tick, N=WS_N):
    """Load ckpt, run forward to target_tick, return final carry + hash."""
    z = np.load(ckpt_path, allow_pickle=False)
    start_tick = int(z["tick"])
    seed = int(z["seed"])
    n_ticks = int(z["n_ticks"])
    if target_tick < start_tick:
        raise ValueError(f"target {target_tick} < ckpt tick {start_tick}")
    if target_tick > n_ticks:
        raise ValueError(f"target {target_tick} > n_ticks {n_ticks}")

    d = {k: z[k] for k in CKPT_KEYS}
    carry = _dict_to_carry(d, N)

    P_np = load_all(params_pkl)
    P, neighbors, _degs = _prepare(P_np, N)
    step_fn = make_step(neighbors, P)
    keys = random.split(random.PRNGKey(seed), n_ticks)
    remaining = keys[start_tick:target_tick]

    @jax.jit
    def run_inner(carry, ks):
        return lax.scan(step_fn, carry, ks)

    carry, _outs = run_inner(carry, remaining)
    jax.block_until_ready(carry)
    d2 = _carry_to_dict(carry)
    return d2, state_hash(d2)
