# engine_v5.py — Phase 2e: unified trust + ultimatum on WS network.
#
# Per tick, each engaged pair plays BOTH games sequentially with the same
# partner, using one shared Q_s2 table over 4 stage-2 action categories:
#   a2 = 0 : self-max  (ulti offer=2, trust send=0)
#   a2 = 1 : self-mild (ulti offer=4, trust send=3)
#   a2 = 2 : fair      (ulti offer=5, trust send=6)
#   a2 = 3 : generous  (ulti offer=7, trust send=9)
# Rewards from the two games sum into one scalar that updates both Q tables
# and the P_hat transition model. Fehr-Schmidt accept/reject is deterministic.
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

P_COMMON     = 0.7
E_ENDOWMENT  = 10.0
TRUST_K      = 3.0
N_ACTIONS    = 4          # stage-2 action categories
ULTI_OFFERS  = jnp.array([2.0, 4.0, 5.0, 7.0])       # a2 -> offer
TRUST_SENDS  = jnp.array([0.0, 3.0, 6.0, 9.0])       # a2 -> send
TRUST_RETURN_MULT = jnp.array([0.0, 1.0, 2.0, 3.0])  # return category -> multiplier of sent

WS_N = 32000
WS_K = 6
WS_P = 0.1
WEALTH_INIT = 10.0

# Coupling constants — all [SYNTH-PRIOR-COEFF]
TAU_BASE       = 2.0
KAPPA_DECAY    = 0.8
DISTRESS_DECAY = 0.95
DISTRESS_PULL  = 0.05
C_BOOST        = 0.3
E_ENGAGE       = 0.5
A_INHIB        = 0.2
T_THREAT       = 0.5
U_UNCERT       = 0.3
SM_DEFECT      = 0.5
SN_STATUS      = 0.3
SP_PUNISH      = 0.3
SAL_GAIN       = 0.2
N_DISTRESS     = 1.0
LAMBDA_NORM_A  = 1.0
GRAD_CLIP      = 100.0
LOGIT_CLIP     = 30.0
REWARD_CLIP    = 100.0

PARAM_NAMES = ['O','C','E','A','N','S_M','S_P','S_N','DASS_Dep','DASS_Anx',
               'DASS_Str','alpha_R','beta_choice','phi_per','w_MB','k_DD',
               'a_inhib','lambda_ineq','theta_T','u_U','eta_vol','omega_social',
               'lambda_norm','s_spite','tau_sens','gamma_sal','kappa_ctx',
               'm_t','D','general_distress']

def load_all(pkl_path):
    df = pd.read_pickle(pkl_path)
    out = {}
    for name in PARAM_NAMES:
        col = name if name in df.columns else None
        if col is None:
            cl = {c.lower(): c for c in df.columns}
            col = cl.get(name.lower())
        if col is None:
            raise KeyError(f"missing {name}")
        out[name] = np.asarray(df[col].values, np.float32)
    return out

def build_ws(n=WS_N, k=WS_K, p=WS_P, seed=0):
    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    degs = np.array([d for _, d in G.degree()])
    K_max = int(degs.max())
    nb = np.full((n, K_max), -1, dtype=np.int32)
    for i, adj in G.adjacency():
        nbrs = sorted(adj.keys())
        nb[i, :len(nbrs)] = nbrs
    return nb, degs

def make_step(neighbors, P):
    N, K = neighbors.shape
    idx = jnp.arange(N)
    E = E_ENDOWMENT
    valid_mask = (neighbors >= 0)
    logits_pick = jnp.where(valid_mask, 0.0, -1e30)

    # ---- static per-agent constants ----
    tau        = TAU_BASE * (0.5 + P['O'])
    alpha_envy = P['D'] + P['lambda_ineq']
    beta_guilt = P['A']
    s_spite    = P['s_spite']
    lr         = P['alpha_R']
    k_DD_raw   = P['k_DD']
    # [COPULA-DEFECT-k_DD] and [TANH-GAMMA-PROVISIONAL] flags from P2d apply
    gamma      = 0.5 + 0.45 * jnp.tanh(k_DD_raw / 3.0)
    beta_soft  = P['beta_choice']
    phi        = P['phi_per']
    w_mb_base  = P['w_MB']
    C_b        = P['C']
    eta_vol    = P['eta_vol']
    ctx        = P['kappa_ctx']
    N_sens     = P['N']
    E_bias     = P['E'] * E_ENGAGE
    a_inhib    = P['a_inhib'] * A_INHIB
    theta_T    = P['theta_T'] * T_THREAT
    u_U        = P['u_U'] * U_UNCERT
    SM_b       = P['S_M'] * SM_DEFECT
    SP_b       = P['S_P'] * SP_PUNISH
    SN_b       = P['S_N'] * SN_STATUS
    omega_soc  = P['omega_social']
    lam_norm   = P['lambda_norm'] * LAMBDA_NORM_A
    sal        = P['gamma_sal'] * SAL_GAIN
    mood       = P['m_t']
    tau_sens   = P['tau_sens']
    baseline_dist = (P['general_distress']
                     + 0.3*P['DASS_Dep'] + 0.3*P['DASS_Anx'] + 0.3*P['DASS_Str']) / 1.9

    def step(carry, tick_key):
        Q_mf, Q_s2, P_hat, wealth, distress, last_a1 = carry
        (k_pick, k_coin, k_a1, k_a2, k_trust, k_role, _) = random.split(tick_key, 7)

        # ---- 1. partner matching ----
        k_idx = random.categorical(k_pick, logits_pick, axis=-1)
        chosen = neighbors[idx, k_idx]
        is_mutual = (chosen[chosen] == idx)
        pair_id = jnp.minimum(idx, chosen)
        coin_all = random.bernoulli(k_coin, 0.5, shape=(N,)).astype(jnp.float32)
        coin_pair = coin_all[pair_id]
        is_lower = idx < chosen
        is_prop_pre = jnp.where(coin_pair == 1.0, is_lower, ~is_lower)
        is_proposer  = is_prop_pre & is_mutual
        is_responder = (~is_prop_pre) & is_mutual

        # ---- 2. arbitration ----
        w_mb_eff = jnp.clip(
            w_mb_base * (1.0 + C_b * C_BOOST) * (1.0 - distress * KAPPA_DECAY),
            0.0, 1.0)

        # ---- 3. stage-1 values (2-step, as v4) ----
        best_s2 = jnp.max(Q_s2, axis=-1)
        Q_MB = jnp.sum(P_hat * best_s2[:, None, :], axis=-1)
        Q_net = w_mb_eff[:, None] * Q_MB + (1.0 - w_mb_eff[:, None]) * Q_mf

        engage_adv = E_bias - a_inhib
        avoid_adv  = theta_T + u_U
        persev = phi * jnp.where(last_a1 == 0, 1.0, 0.0)
        L1 = tau[:, None] * jnp.stack(
            [Q_net[:, 0] + engage_adv + persev, Q_net[:, 1] + avoid_adv], axis=-1)
        L1 = jnp.clip(L1, -LOGIT_CLIP, LOGIT_CLIP)
        a1 = random.categorical(k_a1, L1, axis=-1)

        # ---- 4. transition ----
        common = random.bernoulli(k_role, P_COMMON, shape=(N,))
        s2 = jnp.where(common, a1, 1 - a1)

        # ---- 5. stage-2 action (unified 4 categories) ----
        q_s2_actions = Q_s2[idx, s2, :]  # (N, 4)
        # Dark Triad bias: shift logits toward low-index (selfish) actions
        dark_bias = -SM_b - SN_b + SP_b * 0.0
        action_bonus = jnp.stack(
            [dark_bias, dark_bias * 0.5, jnp.zeros_like(dark_bias), jnp.zeros_like(dark_bias)],
            axis=-1)
        L2 = beta_soft[:, None] * (q_s2_actions + action_bonus)
        L2 = jnp.clip(L2, -LOGIT_CLIP, LOGIT_CLIP)
        a2 = random.categorical(k_a2, L2, axis=-1)  # 0..3

        # ---- 6. ultimatum sub-game ----
        my_offer = ULTI_OFFERS[a2]
        received_offer = my_offer[chosen]

        d_me = 2.0 * received_offer - E
        rp_me = jnp.maximum(d_me, 0.0)
        rn_me = jnp.maximum(-d_me, 0.0)
        my_U_R = received_offer - alpha_envy * rn_me - beta_guilt * rp_me + lam_norm
        my_accept = ((my_U_R >= 0.0) & is_responder).astype(jnp.float32)

        d_p = 2.0 * my_offer - E
        rp_p = jnp.maximum(d_p, 0.0)
        rn_p = jnp.maximum(-d_p, 0.0)
        partner_U_R = my_offer - alpha_envy[chosen] * rn_p - beta_guilt[chosen] * rp_p
        partner_accept = ((partner_U_R >= 0.0) & is_proposer).astype(jnp.float32)

        r_ulti_P = jnp.where(partner_accept >= 1.0, E - my_offer, 0.0)
        r_ulti_R = jnp.where(my_accept >= 1.0, received_offer, 0.0)

        # ---- 7. trust sub-game ----
        # Sender is the OTHER agent (asymmetric to ultimatum: roles are inverted)
        # Each agent is sender when they are not ultimatum-proposer.
        is_trust_sender  = is_responder   # ultimatum responder also sends in trust game
        is_trust_trustee = is_proposer

        my_send = TRUST_SENDS[a2]
        received_send = my_send[chosen]

        # Trustee returns a category based on their own a2
        # (they chose a2 in step 5)
        my_return_mult = TRUST_RETURN_MULT[a2]
        received_return_mult = my_return_mult[chosen]

        # Payout rules:
        #   sender gets E - send + received_back where received_back = mult * send
        #   trustee gets (3*send) - returned_back where returned_back = mult * send
        # Because sent amount is deducted before multiplication (BDM standard):
        #   sender payoff = E - send + K*mult*send
        #   trustee payoff = K*send*(1 - mult/3)  ... (if mult=3 trustee gets 0)
        r_trust_sender = jnp.where(
            is_trust_sender,
            -my_send + TRUST_K * received_return_mult * my_send,
            0.0)
        # Trustee: they receive K*send from the investment pool, then return
        # mult*send to the sender.
        r_trust_trustee = jnp.where(
            is_trust_trustee,
            TRUST_K * received_send * (1.0 - my_return_mult / TRUST_K),
            0.0)

        # ---- 8. total per-agent reward ----
        raw_reward = r_ulti_P + r_ulti_R + r_trust_sender + r_trust_trustee
        reward = jnp.clip(
            raw_reward * omega_soc * (1.0 + sal + 0.05 * mood),
            -REWARD_CLIP, REWARD_CLIP)

        # Engagement gate: non-interacting agents get zero.
        interacting = (a1 == 0) & (a1[chosen] == 0) & is_mutual
        reward = jnp.where(interacting, reward, 0.0)

        # ---- 9. Q updates ----
        Q_s2_prev = Q_s2[idx, s2, a2]
        Q_s2_new_ = jnp.clip(Q_s2_prev + lr * (reward - Q_s2_prev), -GRAD_CLIP, GRAD_CLIP)
        Q_s2_new = Q_s2.at[idx, s2, a2].set(Q_s2_new_)

        Q_mf_prev = Q_mf[idx, a1]
        v_next = jnp.max(Q_mf, axis=-1)
        Q_mf_new_ = jnp.clip(
            Q_mf_prev + lr * (reward + gamma * v_next - Q_mf_prev),
            -GRAD_CLIP, GRAD_CLIP)
        Q_mf_new = Q_mf.at[idx, a1].set(Q_mf_new_)

        # ---- 10. P_hat update ----
        one_hot = jnp.stack([1.0 - s2.astype(jnp.float32),
                             s2.astype(jnp.float32)], axis=-1)
        P_at = P_hat[idx, a1, :]
        P_upd = P_at + eta_vol[:, None] * (one_hot - P_at)
        P_upd = jnp.clip(P_upd, 1e-4, 1.0)
        P_upd = P_upd / jnp.sum(P_upd, axis=-1, keepdims=True)
        P_hat_new = P_hat.at[idx, a1, :].set(P_upd)

        # ---- 11. distress ----
        rejected = is_responder & interacting & (my_U_R < 0.0)
        new_distress = (DISTRESS_DECAY * distress
                        + DISTRESS_PULL * baseline_dist
                        + (1.0 + N_sens * N_DISTRESS) * rejected.astype(jnp.float32) * 0.05)

        # ---- 12. aggregates ----
        new_wealth = jnp.maximum(wealth + reward, 0.0)
        n_pairs = jnp.sum(is_mutual.astype(jnp.float32)) / 2.0
        n_resp = jnp.sum((is_responder & interacting).astype(jnp.float32))
        n_acc  = jnp.sum(my_accept * interacting)
        coop   = jnp.where(n_resp > 0, n_acc / n_resp, 0.0)

        ws_s = jnp.sort(new_wealth)
        tot  = jnp.sum(new_wealth)
        i1   = jnp.arange(1, N + 1, dtype=jnp.float32)
        gini = jnp.where(tot > 0,
                         (2.0 * jnp.sum(i1 * ws_s)) / (N * tot) - (N + 1.0) / N,
                         0.0)

        # Average per-category a2 distribution — proxy for trust/ulti strategy
        onehot_a2 = (a2[:, None] == jnp.arange(N_ACTIONS)[None, :]).astype(jnp.float32)
        mean_a2   = jnp.mean(a2.astype(jnp.float32))

        out = (coop, n_pairs, jnp.mean(new_distress),
               jnp.mean(w_mb_eff), jnp.sum(rejected.astype(jnp.float32)),
               gini, jnp.mean(jnp.abs(Q_net)), jnp.mean(reward), mean_a2)
        return (Q_mf_new, Q_s2_new, P_hat_new, new_wealth, new_distress, a1), out
    return step

def run(params_pkl, n_ticks, seed=42):
    P_np = load_all(params_pkl)
    P = {k: jnp.asarray(v) for k, v in P_np.items()}
    nb, degs = build_ws()
    print(f"[ws] degree min={degs.min()} max={degs.max()} mean={degs.mean():.3f}")
    neighbors = jnp.asarray(nb, jnp.int32)

    Q_mf0  = jnp.zeros((WS_N, 2), jnp.float32)
    Q_s2_0 = jnp.zeros((WS_N, 2, N_ACTIONS), jnp.float32)
    P_hat0 = jnp.full((WS_N, 2, 2), 0.0, jnp.float32)
    P_hat0 = P_hat0.at[:, 0, :].set(jnp.array([P_COMMON, 1.0 - P_COMMON]))
    P_hat0 = P_hat0.at[:, 1, :].set(jnp.array([1.0 - P_COMMON, P_COMMON]))
    wealth0 = jnp.full((WS_N,), WEALTH_INIT, jnp.float32)
    baseline = (P['general_distress'] + 0.3*P['DASS_Dep']
                + 0.3*P['DASS_Anx'] + 0.3*P['DASS_Str']) / 1.9
    last0 = jnp.zeros((WS_N,), jnp.int32)

    step_fn = make_step(neighbors, P)
    keys = random.split(random.PRNGKey(seed), n_ticks)

    @jax.jit
    def run_inner(carry, ks):
        return lax.scan(step_fn, carry, ks)

    carry0 = (Q_mf0, Q_s2_0, P_hat0, wealth0, baseline, last0)
    print(f"[run] {n_ticks} ticks...")
    t0 = time.time()
    final, outs = run_inner(carry0, keys)
    jax.block_until_ready(final)
    wall = time.time() - t0
    print(f"[run] wall = {wall:.3f}s")

    def _n(a): return np.asarray(a)
    return {"wall_s": wall,
            "coop_rate": _n(outs[0]), "n_pairs": _n(outs[1]),
            "mean_distress": _n(outs[2]), "mean_w_mb_eff": _n(outs[3]),
            "defection": _n(outs[4]), "gini": _n(outs[5]),
            "mean_abs_Q_net": _n(outs[6]), "mean_reward": _n(outs[7]),
            "mean_a2": _n(outs[8])}, \
           {"Q_mf": _n(final[0]), "Q_s2": _n(final[1]),
            "P_hat": _n(final[2]), "wealth": _n(final[3]),
            "distress": _n(final[4])}
