# engine.py — Phase 1 environment core (JAX, CUDA)
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import time
import numpy as np
import pandas as pd
import networkx as nx
import jax
import jax.numpy as jnp
from jax import random, lax

E_ENDOWMENT    = 10.0
N_OFFER_LEVELS = 11
KAPPA_DECAY    = 0.8
DISTRESS_DECAY = 0.95
DISTRESS_RISE  = 0.05
WS_N           = 32000
WS_K           = 6
WS_P           = 0.1
WEALTH_INIT    = 10.0

def _find_col(columns, *candidates):
    for c in candidates:
        if c in columns: return c
    cl = [c.lower() for c in columns]
    for cand in candidates:
        for i, c in enumerate(cl):
            if cand.lower() in c: return columns[i]
    return None

def load_params(pkl_path):
    df = pd.read_pickle(pkl_path)
    print(f"[load_params] shape={df.shape}")
    a_col = _find_col(df.columns, 'alpha_R', 'alpha', 'envy', 'alpha_i')
    b_col = _find_col(df.columns, 'beta_choice', 'beta', 'guilt', 'beta_i')
    w_col = _find_col(df.columns, 'w_MB', 'w_mb_base', 'w_mb', 'wmb')
    d_col = _find_col(df.columns, 'general_distress', 'distress', 'm_t', 'mood')
    missing = [n for n, c in [('alpha', a_col), ('beta', b_col),
                              ('w_mb_base', w_col), ('distress', d_col)] if c is None]
    if missing:
        raise KeyError(f"Missing: {missing}. Available: {list(df.columns)}")
    print(f"[load_params] alpha={a_col!r} beta={b_col!r} w_mb={w_col!r} distress={d_col!r}")
    a = np.asarray(df[a_col].values, dtype=np.float32)
    b = np.asarray(df[b_col].values, dtype=np.float32)
    w = np.asarray(df[w_col].values, dtype=np.float32)
    d = np.asarray(df[d_col].values, dtype=np.float32)
    for name, arr in [("alpha", a), ("beta", b), ("w_mb_base", w), ("distress", d)]:
        assert arr.shape == (WS_N,), f"{name} shape {arr.shape} != ({WS_N},)"
    return a, b, w, d

def build_ws_adjacency(n=WS_N, k=WS_K, p=WS_P, seed=0):
    print(f"[build_ws] Watts-Strogatz n={n} k={k} p={p}")
    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    degrees = np.array([d for _, d in G.degree()])
    K_max = int(degrees.max()); K_min = int(degrees.min())
    print(f"[build_ws] degree min={K_min} max={K_max} mean={degrees.mean():.3f}")
    nb = np.full((n, K_max), -1, dtype=np.int32)
    for i, adj in G.adjacency():
        nbrs = sorted(adj.keys())
        nb[i, :len(nbrs)] = nbrs
    return nb, K_max

def make_step(neighbors, alpha, beta, w_mb_base, mu_alpha, mu_beta):
    N, K = neighbors.shape
    idx = jnp.arange(N)
    offers = jnp.arange(N_OFFER_LEVELS, dtype=jnp.float32)
    E = E_ENDOWMENT

    valid_mask = (neighbors >= 0)
    logits = jnp.where(valid_mask, 0.0, -1e30)

    diff      = 2.0 * offers - E
    relu_pos  = jnp.maximum(diff, 0.0)
    relu_neg  = jnp.maximum(-diff, 0.0)
    u_r_pop   = offers - mu_alpha * relu_neg - mu_beta * relu_pos
    accept_mask_pop = (u_r_pop >= 0.0).astype(jnp.float32)

    alpha_col = alpha[:, None]; beta_col = beta[:, None]
    u_p_all   = (E - offers[None, :]) - alpha_col * relu_pos[None, :] - beta_col * relu_neg[None, :]
    u_p_masked = jnp.where(accept_mask_pop[None, :] >= 1.0, u_p_all, -1e9)

    def step(carry, tick_key):
        wealth, distress, last_accept = carry
        k_pick, k_coin = random.split(tick_key, 2)

        k_idx  = random.categorical(k_pick, logits, axis=-1)
        chosen = neighbors[idx, k_idx]

        partner_of_partner = chosen[chosen]
        is_mutual = (partner_of_partner == idx)

        pair_id  = jnp.minimum(idx, chosen)
        coin_all = random.bernoulli(k_coin, 0.5, shape=(N,)).astype(jnp.float32)
        coin_pair = coin_all[pair_id]
        is_lower = idx < chosen
        is_prop_pre = jnp.where(coin_pair == 1.0, is_lower, ~is_lower)
        is_proposer  = is_prop_pre & is_mutual
        is_responder = (~is_prop_pre) & is_mutual

        chosen_offer_idx = jnp.argmax(u_p_masked, axis=1)
        chosen_offer     = offers[chosen_offer_idx]
        offer_received   = chosen_offer[chosen]

        diff_at     = 2.0 * offer_received - E
        relu_neg_at = jnp.maximum(-diff_at, 0.0)
        relu_pos_at = jnp.maximum(diff_at, 0.0)
        u_r_true    = offer_received - alpha * relu_neg_at - beta * relu_pos_at
        accept_raw  = (u_r_true >= 0.0).astype(jnp.float32)
        accept_dec  = accept_raw * is_responder.astype(jnp.float32)

        accept_of_partner = accept_dec[chosen]
        payout_P = jnp.where(is_proposer & (accept_of_partner >= 1.0), E - chosen_offer, 0.0)
        payout_R = jnp.where(is_responder & (accept_dec >= 1.0), offer_received, 0.0)
        new_wealth = wealth + payout_P + payout_R

        new_distress = jnp.where(
            is_responder,
            DISTRESS_DECAY * distress + DISTRESS_RISE * (1.0 - accept_dec),
            distress,
        )
        new_last_accept = jnp.where(is_proposer, accept_of_partner, last_accept)
        w_mb_eff = w_mb_base * (1.0 - new_distress * KAPPA_DECAY)

        n_resp   = jnp.sum(is_responder.astype(jnp.float32))
        n_accept = jnp.sum(accept_dec)
        n_mutual = jnp.sum(is_mutual.astype(jnp.float32))
        coop_rate = jnp.where(n_resp > 0, n_accept / n_resp, 0.0)

        ws_sorted = jnp.sort(new_wealth)
        total     = jnp.sum(new_wealth)
        i_plus_1  = jnp.arange(1, N + 1, dtype=jnp.float32)
        gini = jnp.where(
            total > 0,
            (2.0 * jnp.sum(i_plus_1 * ws_sorted)) / (N * total) - (N + 1.0) / N,
            0.0,
        )

        out = (coop_rate, n_mutual / 2.0, jnp.mean(new_distress),
               jnp.mean(w_mb_eff),
               jnp.sum(is_responder.astype(jnp.float32) * (1.0 - accept_dec)),
               gini, n_accept, n_resp)
        return (new_wealth, new_distress, new_last_accept), out
    return step

def run_simulation(params_pkl, n_ticks, seed=42):
    alpha_np, beta_np, w_np, d_np = load_params(params_pkl)
    neighbors_np, K_max = build_ws_adjacency()
    neighbors   = jnp.asarray(neighbors_np, dtype=jnp.int32)
    alpha       = jnp.asarray(alpha_np)
    beta        = jnp.asarray(beta_np)
    w_mb_base   = jnp.asarray(w_np)
    distress0   = jnp.asarray(d_np)
    mu_alpha    = jnp.mean(alpha); mu_beta = jnp.mean(beta)
    print(f"[run_sim] mu_alpha={float(mu_alpha):.4f}  mu_beta={float(mu_beta):.4f}")

    wealth0 = jnp.full((WS_N,), WEALTH_INIT, dtype=jnp.float32)
    last0   = jnp.ones((WS_N,), dtype=jnp.float32)

    step_fn = make_step(neighbors, alpha, beta, w_mb_base, mu_alpha, mu_beta)
    tick_keys = random.split(random.PRNGKey(seed), n_ticks)

    @jax.jit
    def full_run(carry, keys):
        return lax.scan(step_fn, carry, keys)

    carry0 = (wealth0, distress0, last0)
    print(f"[run_sim] running {n_ticks} ticks...")
    t0 = time.time()
    final_carry, outs = full_run(carry0, tick_keys)
    jax.block_until_ready(final_carry)
    wall = time.time() - t0
    print(f"[run_sim] wall_time_s = {wall:.3f}")

    fw, fd, fl = final_carry
    metrics = {
        'coop_rate':       np.asarray(outs[0]),
        'n_mutual_pairs':  np.asarray(outs[1]),
        'mean_distress':   np.asarray(outs[2]),
        'mean_w_mb_eff':   np.asarray(outs[3]),
        'defection_count': np.asarray(outs[4]),
        'gini':            np.asarray(outs[5]),
        'n_accept':        np.asarray(outs[6]),
        'n_responder':     np.asarray(outs[7]),
    }
    final_state = {
        'wealth':      np.asarray(fw),
        'distress':    np.asarray(fd),
        'last_accept': np.asarray(fl),
    }
    return metrics, final_state, wall
