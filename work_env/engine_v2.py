# engine_v2.py — Phase 2a dual-system social agent
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

E_ENDOWMENT  = 10.0
P_COMMON     = 0.7
FAIR_OFFER   = 5.0
SELFISH_OFFER= 2.0
KAPPA_DECAY  = 0.8
DISTRESS_DECAY = 0.95
DISTRESS_RISE  = 0.05
WS_N = 32000
WS_K = 6
WS_P = 0.1
WEALTH_INIT = 10.0

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
    a_col = _find_col(df.columns, 'alpha_R', 'alpha')
    b_col = _find_col(df.columns, 'beta_choice', 'beta')
    w_col = _find_col(df.columns, 'w_MB', 'w_mb_base', 'w_mb')
    d_col = _find_col(df.columns, 'general_distress', 'distress')
    for n, c in [('alpha', a_col), ('beta', b_col), ('w_mb', w_col), ('distress', d_col)]:
        if c is None:
            raise KeyError(f"missing {n}; cols={list(df.columns)}")
    return (np.asarray(df[a_col].values, np.float32),
            np.asarray(df[b_col].values, np.float32),
            np.asarray(df[w_col].values, np.float32),
            np.asarray(df[d_col].values, np.float32))

def build_ws_adjacency(n=WS_N, k=WS_K, p=WS_P, seed=0):
    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    degs = np.array([d for _, d in G.degree()])
    K_max = int(degs.max())
    nb = np.full((n, K_max), -1, dtype=np.int32)
    for i, adj in G.adjacency():
        nbrs = sorted(adj.keys())
        nb[i, :len(nbrs)] = nbrs
    return nb, K_max, int(degs.min()), K_max

def make_step(neighbors, alpha, beta, w_mb_base):
    N, K = neighbors.shape
    idx = jnp.arange(N)
    E = E_ENDOWMENT
    valid_mask = (neighbors >= 0)
    logits = jnp.where(valid_mask, 0.0, -1e30)

    def step(carry, tick_key):
        Q_mf, Q_td, wealth, distress, _last = carry
        k_pick, k_coin, k_a1, k_trans, k_a2 = random.split(tick_key, 5)

        # -- Partner selection (mutual) --
        k_idx  = random.categorical(k_pick, logits, axis=-1)
        chosen = neighbors[idx, k_idx]
        is_mutual = (chosen[chosen] == idx)
        pair_id = jnp.minimum(idx, chosen)
        coin_all = random.bernoulli(k_coin, 0.5, shape=(N,)).astype(jnp.float32)
        coin_pair = coin_all[pair_id]
        is_lower = idx < chosen
        is_prop_pre = jnp.where(coin_pair == 1.0, is_lower, ~is_lower)
        is_proposer  = is_prop_pre & is_mutual
        is_responder = (~is_prop_pre) & is_mutual

        # -- w_mb_eff from current distress --
        w_mb_eff = jnp.clip(w_mb_base * (1.0 - distress * KAPPA_DECAY), 0.0, 1.0)

        # -- Model-based stage-1 values --
        m = jnp.max(Q_td, axis=-1)                # (N, 2)
        Q_mb = jnp.stack([
            P_COMMON * m[:, 0] + (1 - P_COMMON) * m[:, 1],
            P_COMMON * m[:, 1] + (1 - P_COMMON) * m[:, 0],
        ], axis=-1)                                # (N, 2)

        # -- Arbitration --
        Q_net = w_mb_eff[:, None] * Q_mb + (1 - w_mb_eff[:, None]) * Q_mf

        # -- Stage-1 action --
        a1 = random.categorical(k_a1, beta[:, None] * Q_net, axis=-1)     # (N,)

        # -- Transition --
        common = random.bernoulli(k_trans, P_COMMON, shape=(N,))
        s2 = jnp.where(common, a1, 1 - a1)

        # -- Stage-2 action --
        q_td_s2 = Q_td[idx, s2, :]                # (N, 2)
        a2 = random.categorical(k_a2, beta[:, None] * q_td_s2, axis=-1)   # (N,)

        # -- Interaction --
        partner_a1 = a1[chosen]
        partner_a2 = a2[chosen]
        both_engaged = (a1 == 0) & (partner_a1 == 0) & is_mutual

        my_offer      = jnp.where(a2 == 0, FAIR_OFFER, SELFISH_OFFER)
        partner_offer = jnp.where(partner_a2 == 0, FAIR_OFFER, SELFISH_OFFER)

        # Responder's decision (own α, β)
        diff_me = 2.0 * partner_offer - E
        relu_pos_me = jnp.maximum(diff_me, 0.0)
        relu_neg_me = jnp.maximum(-diff_me, 0.0)
        my_U_R = partner_offer - alpha * relu_neg_me - beta * relu_pos_me
        my_accept = (my_U_R >= 0.0) & is_responder & both_engaged

        # Partner's decision (partner's α, β)
        diff_partner = 2.0 * my_offer - E
        relu_pos_p = jnp.maximum(diff_partner, 0.0)
        relu_neg_p = jnp.maximum(-diff_partner, 0.0)
        partner_U_R = my_offer - alpha[chosen] * relu_neg_p - beta[chosen] * relu_pos_p
        partner_accept = (partner_U_R >= 0.0) & is_proposer & both_engaged

        reward_proposer  = jnp.where(partner_accept, E - my_offer, 0.0)
        reward_responder = jnp.where(my_accept, partner_offer, 0.0)
        reward = reward_proposer + reward_responder

        # -- Q updates --
        q_td_cur = Q_td[idx, s2, a2]
        Q_td_new = Q_td.at[idx, s2, a2].set(q_td_cur + alpha * (reward - q_td_cur))
        q_mf_cur = Q_mf[idx, a1]
        Q_mf_new = Q_mf.at[idx, a1].set(q_mf_cur + alpha * (reward - q_mf_cur))

        # -- Distress update --
        rejected = is_responder & both_engaged & (my_U_R < 0.0)
        new_distress = DISTRESS_DECAY * distress + DISTRESS_RISE * rejected.astype(jnp.float32)

        # -- Wealth --
        new_wealth = wealth + reward

        # -- w_mb_eff after update --
        new_w_mb_eff = jnp.clip(w_mb_base * (1.0 - new_distress * KAPPA_DECAY), 0.0, 1.0)

        # -- Aggregates --
        n_resp_decisions = jnp.sum((is_responder & both_engaged).astype(jnp.float32))
        n_accepts = jnp.sum(my_accept.astype(jnp.float32))
        coop_rate = jnp.where(n_resp_decisions > 0, n_accepts / n_resp_decisions, 0.0)

        ws_sorted = jnp.sort(new_wealth)
        total = jnp.sum(new_wealth)
        i_plus_1 = jnp.arange(1, N + 1, dtype=jnp.float32)
        gini = jnp.where(total > 0,
                         (2.0 * jnp.sum(i_plus_1 * ws_sorted)) / (N * total) - (N + 1.0) / N,
                         0.0)

        out = (coop_rate,
               jnp.sum(is_mutual.astype(jnp.float32)) / 2.0,
               jnp.mean(new_distress),
               jnp.mean(new_w_mb_eff),
               jnp.sum(rejected.astype(jnp.float32)),
               gini,
               jnp.mean(jnp.abs(Q_net)),
               jnp.mean(reward))
        return (Q_mf_new, Q_td_new, new_wealth, new_distress, a1), out
    return step

def run_simulation(params_pkl, n_ticks, seed=42):
    alpha_np, beta_np, w_np, d_np = load_params(params_pkl)
    nb_np, K_max, dmin, dmax = build_ws_adjacency()
    print(f"[build_ws] degree min={dmin} max={dmax}")
    neighbors = jnp.asarray(nb_np, jnp.int32)
    alpha     = jnp.asarray(alpha_np)
    beta      = jnp.asarray(beta_np)
    w_mb_base = jnp.asarray(w_np)
    distress0 = jnp.asarray(d_np)

    Q_mf0 = jnp.zeros((WS_N, 2), dtype=jnp.float32)
    Q_td0 = jnp.zeros((WS_N, 2, 2), dtype=jnp.float32)
    wealth0 = jnp.full((WS_N,), WEALTH_INIT, dtype=jnp.float32)
    last0 = jnp.zeros((WS_N,), dtype=jnp.int32)

    step_fn = make_step(neighbors, alpha, beta, w_mb_base)
    tick_keys = random.split(random.PRNGKey(seed), n_ticks)

    @jax.jit
    def full_run(carry, keys):
        return lax.scan(step_fn, carry, keys)

    carry0 = (Q_mf0, Q_td0, wealth0, distress0, last0)
    print(f"[run_sim] running {n_ticks} ticks...")
    t0 = time.time()
    final_carry, outs = full_run(carry0, tick_keys)
    jax.block_until_ready(final_carry)
    wall = time.time() - t0
    print(f"[run_sim] wall_time_s = {wall:.3f}")

    fQmf, fQtd, fW, fD, fLast = final_carry
    metrics = {
        'coop_rate':       np.asarray(outs[0]),
        'n_mutual_pairs':  np.asarray(outs[1]),
        'mean_distress':   np.asarray(outs[2]),
        'mean_w_mb_eff':   np.asarray(outs[3]),
        'defection_count': np.asarray(outs[4]),
        'gini':            np.asarray(outs[5]),
        'mean_abs_Q_net':  np.asarray(outs[6]),
        'mean_reward':     np.asarray(outs[7]),
    }
    final_state = {
        'Q_mf':    np.asarray(fQmf),
        'Q_td':    np.asarray(fQtd),
        'wealth':  np.asarray(fW),
        'distress':np.asarray(fD),
    }
    return metrics, final_state, wall
