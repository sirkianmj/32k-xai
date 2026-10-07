# engine_v4.py — Phase 2d: real dual-system agent on the two-step task.
#
# Replaces the heuristic "role-weighted value" of engine_v3 with the
# spec's Part V.2 MB module: Q_MB(s,a) = sum_s' P_hat(s'|s,a) max_a' Q_s2(s',a')
# where P_hat is a per-agent learned transition tensor.
#
# k_DD role changes: instead of an ad-hoc learning-rate attenuator (v3, wrong),
# k_DD now sets the discount factor gamma = 1/(1+clip(k_DD,0,10)). This is
# the scientifically correct role for a delay-discount parameter in a task
# with temporal credit assignment.
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

# -------- task structure --------
P_COMMON     = 0.7            # task design (spec §2.3, G8)
E_ENDOWMENT  = 10.0
N_OFFER      = 11
WS_N         = 32000
WS_K         = 6
WS_P         = 0.1
WEALTH_INIT  = 10.0

# -------- coupling constants (all [SYNTH-PRIOR-COEFF]) --------
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
GAMMA_K        = 0.1          # gamma = 1/(1 + GAMMA_K * clip(k_DD,0,10))
GRAD_CLIP      = 100.0
LOGIT_CLIP     = 30.0
REWARD_CLIP    = 50.0

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
    offers = jnp.arange(N_OFFER, dtype=jnp.float32)
    valid_mask = (neighbors >= 0)
    logits_pick = jnp.where(valid_mask, 0.0, -1e30)

    # ---- static per-agent constants (evaluated once) ----
    tau          = TAU_BASE * (0.5 + P['O'])                  # Openness -> exploration
    alpha_envy   = P['D'] + P['lambda_ineq']                   # Dark Core + inequity -> envy
    beta_guilt   = P['A']                                       # Agreeableness -> guilt
    s_spite      = P['s_spite']
    lr           = P['alpha_R']
    # gamma from k_DD. The copula's k_DD column has mean -5.17 and range
    # [-9.1, +3.2], violating the spec's k_DD > 0 constraint. This is a
    # G2 copula defect, flagged [COPULA-DEFECT-k_DD]. Until G2 is re-run,
    # use a monotone tanh transform that guarantees a real spread of
    # discount factors regardless of the raw input scale.
    # [SYNTH-PRIOR-COEFF]
    k_DD_raw   = P['k_DD']
    gamma      = 0.5 + 0.45 * jnp.tanh(k_DD_raw / 3.0)
    beta_soft    = P['beta_choice']
    phi          = P['phi_per']
    w_mb_base    = P['w_MB']
    C_b          = P['C']
    eta_vol      = P['eta_vol']
    ctx          = P['kappa_ctx']
    N_sens       = P['N']
    E_bias       = P['E'] * E_ENGAGE
    a_inhib      = P['a_inhib'] * A_INHIB
    theta_T      = P['theta_T'] * T_THREAT
    u_U          = P['u_U'] * U_UNCERT
    SM_b         = P['S_M'] * SM_DEFECT
    SP_b         = P['S_P'] * SP_PUNISH
    SN_b         = P['S_N'] * SN_STATUS
    omega_soc    = P['omega_social']
    lam_norm     = P['lambda_norm'] * LAMBDA_NORM_A
    sal          = P['gamma_sal'] * SAL_GAIN
    mood         = P['m_t']
    tau_sens     = P['tau_sens']
    baseline_dist = (P['general_distress']
                     + 0.3*P['DASS_Dep'] + 0.3*P['DASS_Anx'] + 0.3*P['DASS_Str']) / 1.9

    # Fehr-Schmidt offer tables
    diff = 2.0 * offers - E
    relu_p_off = jnp.maximum(diff, 0.0)
    relu_n_off = jnp.maximum(-diff, 0.0)
    # log-prob table for categorical on offers (deterministic argmax via Gumbel-like
    # is unnecessary; tau_sens weights the argmax)
    tau_sens_w = 0.5 + 3.0 * tau_sens

    def step(carry, tick_key):
        Q_mf, Q_s2, P_hat, wealth, distress, last_a1 = carry
        (k_pick, k_coin, k_a1, k_a2, k_role, _) = random.split(tick_key, 6)

        # ---------------- 1. partner matching (mutual pick) ----------------
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

        # ---------------- 2. arbitration weight ----------------
        w_mb_eff = jnp.clip(
            w_mb_base * (1.0 + C_b * C_BOOST) * (1.0 - distress * KAPPA_DECAY),
            0.0, 1.0)   # (N,)

        # ---------------- 3. stage-1 values ----------------
        # Model-based: Bellman over learned transition tensor P_hat[a1, s2]
        # Q_MB[a1] = sum_s2 P_hat[a1, s2] * max_a2 Q_s2[s2, :]
        best_s2 = jnp.max(Q_s2, axis=-1)             # (N, 2)
        Q_MB = jnp.sum(P_hat * best_s2[:, None, :], axis=-1)  # (N, 2)

        # Model-free: current Q_mf over stage-1 actions
        Q_net = w_mb_eff[:, None] * Q_MB + (1.0 - w_mb_eff[:, None]) * Q_mf

        # Add spec-part-V biases
        # engage_bonus: extraversion drives toward engage, a_inhib toward avoid
        # avoid_bonus:  threat sensitivity + uncertainty aversion
        engage_adv = E_bias - a_inhib
        avoid_adv  = theta_T + u_U
        persev = phi * jnp.where(last_a1 == 0, 1.0, 0.0)
        L1 = tau[:, None] * jnp.stack(
            [Q_net[:, 0] + engage_adv + persev, Q_net[:, 1] + avoid_adv], axis=-1)
        L1 = jnp.clip(L1, -LOGIT_CLIP, LOGIT_CLIP)
        a1 = random.categorical(k_a1, L1, axis=-1)   # 0 = engage, 1 = avoid

        # ---------------- 4. transition ----------------
        # Standard two-step: common (P_COMMON) means s2 = a1, rare -> 1-a1
        common = random.bernoulli(k_role, P_COMMON, shape=(N,))
        s2 = jnp.where(common, a1, 1 - a1)

        # ---------------- 5. stage-2 action ----------------
        # In this environment, stage-2 action a2 = offer level bias category.
        # 0 = cooperate (fair), 1 = defect (selfish).
        q_coop = Q_s2[idx, s2, 0]
        q_def  = Q_s2[idx, s2, 1]
        strategic = SM_b + SN_b       # Dark Triad bonus for defection
        punish_pull = SP_b            # psychopathy reduces punishment sensitivity
        L2 = jnp.stack(
            [q_coop,
             q_def + strategic - punish_pull * jnp.abs(q_def)], axis=-1)
        L2 = beta_soft[:, None] * L2
        L2 = jnp.clip(L2, -LOGIT_CLIP, LOGIT_CLIP)
        a2 = random.categorical(k_a2, L2, axis=-1)   # 0 = cooperate, 1 = defect

        # ---------------- 6. Fehr-Schmidt offer argmax ----------------
        U_P = (E - offers[None, :]) \
              - alpha_envy[:, None] * relu_p_off[None, :] \
              - beta_guilt[:, None] * relu_n_off[None, :] \
              - s_spite[:, None] * relu_p_off[None, :] * 0.5
        # cooperate shifts offer toward fair (5); defect toward selfish (2)
        offer_mean = jnp.where(a2 == 0, 5.0, 2.0)
        U_P = U_P + tau_sens_w[:, None] * (-jnp.abs(offers[None, :] - offer_mean[:, None]))
        my_offer = offers[jnp.argmax(U_P, axis=-1)]
        received_offer = my_offer[chosen]

        # ---------------- 7. interaction and reward ----------------
        # Engagement requires a1=0 for both partners and mutual pick.
        my_engaged      = (a1 == 0)
        partner_engaged = (a1[chosen] == 0)
        interacting = my_engaged & partner_engaged & is_mutual

        # Responder-side utility given received offer
        d_me   = 2.0 * received_offer - E
        rp_me  = jnp.maximum(d_me, 0.0)
        rn_me  = jnp.maximum(-d_me, 0.0)
        my_U_R = received_offer - alpha_envy * rn_me - beta_guilt * rp_me + lam_norm
        my_accept = ((my_U_R >= 0.0) & is_responder & interacting).astype(jnp.float32)

        # Partner's utility given my offer
        d_p   = 2.0 * my_offer - E
        rp_p  = jnp.maximum(d_p, 0.0)
        rn_p  = jnp.maximum(-d_p, 0.0)
        partner_U_R = my_offer - alpha_envy[chosen] * rn_p - beta_guilt[chosen] * rp_p
        partner_accept = ((partner_U_R >= 0.0) & is_proposer & interacting).astype(jnp.float32)

        r_P = jnp.where(partner_accept >= 1.0, E - my_offer, 0.0)
        r_R = jnp.where(my_accept >= 1.0, received_offer, 0.0)
        raw_reward = r_P + r_R
        reward = jnp.clip(raw_reward * omega_soc * (1.0 + sal + 0.05 * mood),
                          -REWARD_CLIP, REWARD_CLIP)

        # ---------------- 8. TD updates ----------------
        # Stage-2 is terminal: Q_s2[s2, a2] <- Q_s2[s2, a2] + lr*(r - Q_s2[s2, a2])
        Q_s2_prev = Q_s2[idx, s2, a2]
        Q_s2_new_sa = jnp.clip(Q_s2_prev + lr * (reward - Q_s2_prev),
                               -GRAD_CLIP, GRAD_CLIP)
        Q_s2_new = Q_s2.at[idx, s2, a2].set(Q_s2_new_sa)

        # Stage-1 MF bootstraps: Q_mf[a1] <- Q_mf[a1] + lr*(r + gamma*max_a Q_mf - Q_mf[a1])
        Q_mf_prev = Q_mf[idx, a1]
        v_next_mf = jnp.max(Q_mf, axis=-1)
        Q_mf_new_a = jnp.clip(
            Q_mf_prev + lr * (reward + gamma * v_next_mf - Q_mf_prev),
            -GRAD_CLIP, GRAD_CLIP)
        Q_mf_new = Q_mf.at[idx, a1].set(Q_mf_new_a)

        # ---------------- 9. transition model update ----------------
        # P_hat[a1, s2_obs] <- P_hat[a1, s2_obs] + eta_vol*(1 - P_hat[a1, s2_obs])
        one_hot = jnp.stack([1.0 - s2.astype(jnp.float32),
                             s2.astype(jnp.float32)], axis=-1)   # (N, 2)
        # index P_hat at [a1]
        P_at = P_hat[idx, a1, :]                                  # (N, 2)
        P_upd = P_at + eta_vol[:, None] * (one_hot - P_at)
        P_upd = jnp.clip(P_upd, 1e-4, 1.0)
        P_upd = P_upd / jnp.sum(P_upd, axis=-1, keepdims=True)
        P_hat_new = P_hat.at[idx, a1, :].set(P_upd)

        # ---------------- 10. distress update ----------------
        rejected = is_responder & interacting & (my_U_R < 0.0)
        new_distress = (DISTRESS_DECAY * distress
                        + DISTRESS_PULL * baseline_dist
                        + (1.0 + N_sens * N_DISTRESS) * rejected.astype(jnp.float32) * 0.05)

        # ---------------- 11. wealth and aggregates ----------------
        new_wealth = jnp.maximum(wealth + reward, 0.0)
        n_pairs = jnp.sum(is_mutual.astype(jnp.float32)) / 2.0
        n_resp  = jnp.sum((is_responder & interacting).astype(jnp.float32))
        n_acc   = jnp.sum(my_accept)
        coop    = jnp.where(n_resp > 0, n_acc / n_resp, 0.0)

        ws_s = jnp.sort(new_wealth)
        tot  = jnp.sum(new_wealth)
        i1   = jnp.arange(1, N + 1, dtype=jnp.float32)
        gini = jnp.where(tot > 0,
                         (2.0 * jnp.sum(i1 * ws_s)) / (N * tot) - (N + 1.0) / N,
                         0.0)

        out = (coop, n_pairs, jnp.mean(new_distress),
               jnp.mean(w_mb_eff), jnp.sum(rejected.astype(jnp.float32)),
               gini, jnp.mean(jnp.abs(Q_net)), jnp.mean(reward))
        return (Q_mf_new, Q_s2_new, P_hat_new, new_wealth, new_distress, a1), out
    return step

def run(params_pkl, n_ticks, seed=42):
    P_np = load_all(params_pkl)
    P = {k: jnp.asarray(v) for k, v in P_np.items()}
    nb, degs = build_ws()
    print(f"[ws] degree min={degs.min()} max={degs.max()} mean={degs.mean():.3f}")
    neighbors = jnp.asarray(nb, jnp.int32)

    Q_mf0  = jnp.zeros((WS_N, 2), jnp.float32)
    Q_s2_0 = jnp.zeros((WS_N, 2, 2), jnp.float32)   # [s2, a2]
    # P_hat initialized at task design P_COMMON to avoid early-stage misinference
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
            "mean_abs_Q_net": _n(outs[6]), "mean_reward": _n(outs[7])}, \
           {"Q_mf": _n(final[0]), "Q_s2": _n(final[1]),
            "P_hat": _n(final[2]), "wealth": _n(final[3]),
            "distress": _n(final[4])}
