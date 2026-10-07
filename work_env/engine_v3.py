# engine_v3.py — Phase 2c: deep agent per Master Spec Part V.
#
# Every one of the 30 copula columns is read at least once per tick.
# Coefficients tagged [SYNTH-PRIOR-COEFF] are provisional tunings, NOT
# derived from data; they must be swept in Phase 6. Task-specific
# parameters (k_DD, a_inhib) do not have a natural role in a one-shot
# ultimatum; they are wired via documented heuristics and flagged.
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

E_ENDOWMENT   = 10.0
N_OFFER       = 11
KAPPA_DECAY   = 0.8
DISTRESS_DECAY= 0.95
DISTRESS_PULL = 0.05
TAU_BASE      = 2.0
WS_N = 32000
WS_K = 6
WS_P = 0.1
WEALTH_INIT   = 10.0

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

    # ---------- per-agent static constants (evaluated once, before scan) ----------
    # O   -> exploration temperature amplifier            [COEFF ok]
    tau = TAU_BASE * (0.5 + P['O'])
    # D   -> Dark Core feeds Fehr-Schmidt alpha_envy       [spec Part V.1]
    alpha_envy = P['D']
    # A   -> Agreeableness feeds Fehr-Schmidt beta_guilt   [spec Part V.1]
    beta_guilt = P['A']
    # lambda_ineq further shifts alpha_envy in the FS offer argument  [spec Part V.1]
    alpha_eff = alpha_envy + P['lambda_ineq']
    # s_spite: spite term in U_P                            [COEFF 0.5]
    s_spite = P['s_spite']
    # alpha_R -> learning rate                              [spec Part V.2]
    lr = P['alpha_R']
    # k_DD -> delay discount. No time delay in one-shot ultimatum;
    #         wired as effective LR attenuation via sigmoid (bounded [0,1]).
    #         [TASK-MISMATCH] flagged; sigmoid form prevents non-contractive
    #         update when copula produces negative k_DD.
    k_DD = P['k_DD']
    lr_eff = jnp.clip(lr * (1.0 / (1.0 + jnp.exp(k_DD))), 0.01, 0.9)
    # beta_choice -> inverse temperature on softmax          [spec Part V.3]
    beta_soft = P['beta_choice']
    # phi_per -> perseveration bonus on previous engage     [spec Part V.2]
    phi = P['phi_per']
    # w_MB  -> arbitration                                    [spec Part V.2]
    w_mb_base = P['w_MB']
    # C     -> Conscientiousness boosts w_mb_eff              [COEFF 0.3]
    C_boost = P['C']
    # eta_vol -> P_role learning rate                         [COEFF natural]
    eta_vol = P['eta_vol']
    # kappa_ctx -> blend learned vs uniform P_role prior      [COEFF natural]
    ctx = P['kappa_ctx']
    # N       -> neuroticism amplifies distress jumps         [COEFF 1.0]
    N_sens = P['N']
    # E       -> extraversion biases stage-1 toward engage    [COEFF 0.5]
    E_bias = P['E']
    # a_inhib -> inhibitory control lowers engage bonus       [COEFF 0.2, TASK-MISMATCH]
    a_inhib = P['a_inhib']
    # theta_T -> threat sensitivity adds avoid bias           [COEFF 0.5]
    theta_T = P['theta_T']
    # u_U     -> uncertainty aversion adds avoid bias          [COEFF 0.3]
    u_U = P['u_U']
    # S_M/S_P/S_N -> Dark Triad specific factors on stage-2   [COEFFs 0.5, 0.3, 0.3]
    SM_b = P['S_M']; SP_b = P['S_P']; SN_b = P['S_N']
    # omega_social -> multiplicative reward weight            [COEFF natural]
    omega_soc = P['omega_social']
    # lambda_norm -> norm bonus on responder accept           [COEFF 1.0]
    lam_norm = P['lambda_norm']
    # gamma_sal -> salience on reward                         [COEFF 0.2]
    sal = P['gamma_sal']
    # m_t -> mood offset on reward                            [COEFF 0.05]
    mood = P['m_t']
    # tau_sens -> sensory precision on offer argmax           [COEFF 3.0]
    tau_sens = P['tau_sens']
    # DASS bifactor -> baseline distress target               [spec Part II.2]
    baseline_dist = (P['general_distress']
                     + 0.3*P['DASS_Dep']
                     + 0.3*P['DASS_Anx']
                     + 0.3*P['DASS_Str']) / 1.9

    # FS offer-level tables (index over 11 discrete offers)
    diff = 2.0 * offers - E
    relu_p_off = jnp.maximum(diff, 0.0)      # max(2o - E, 0)
    relu_n_off = jnp.maximum(-diff, 0.0)     # max(E - 2o, 0)

    def step(carry, tick_key):
        Q_mf, Q_s2, P_role, wealth, distress, last_a1 = carry
        (k_pick, k_coin, k_a1, k_role, k_a2, _) = random.split(tick_key, 6)

        # ---------------- 1. partner matching ----------------
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

        # ---------------- 2. Model-based stage-1 value ----------------
        # w_mb_eff: C (conscientiousness) boosts, distress attenuates.
        w_mb_eff = jnp.clip(
            w_mb_base * (1.0 + C_boost * 0.3) * (1.0 - distress * KAPPA_DECAY),
            0.0, 1.0)  # (N,)

        # P_role is learned; ctx blends toward uniform prior
        p_prop = ctx * P_role[:, 0] + (1.0 - ctx) * 0.5
        p_resp = ctx * P_role[:, 1] + (1.0 - ctx) * 0.5
        best_prop = jnp.maximum(Q_s2[:, 0], Q_s2[:, 1])   # proposer coop/def
        best_resp = jnp.maximum(Q_s2[:, 2], Q_s2[:, 3])   # responder acc/rej
        Q_mb = p_prop * best_prop + p_resp * best_resp    # (N,)

        # Stage-1 action values: engage = w*MB + (1-w)*MF[engage]
        #                       avoid  =                 MF[avoid]
        Q_net_engage = w_mb_eff * Q_mb + (1.0 - w_mb_eff) * Q_mf[:, 0]
        Q_net_avoid  = Q_mf[:, 1]

        engage_bonus = E_bias * 0.5 - a_inhib * 0.2
        avoid_bonus  = theta_T * 0.5 + u_U * 0.3
        persev = phi * jnp.where(last_a1 == 0, 1.0, 0.0)
        L_eng = tau * (Q_net_engage + engage_bonus + persev)
        L_avd = tau * (Q_net_avoid  + avoid_bonus)
        L1 = jnp.clip(jnp.stack([L_eng, L_avd], axis=-1), -30.0, 30.0)
        a1 = random.categorical(k_a1, L1, axis=-1)

        # ---------------- 3. engagement condition ----------------
        my_engaged      = (a1 == 0)
        partner_engaged = (a1[chosen] == 0)
        interacting = my_engaged & partner_engaged & is_mutual

        # ---------------- 4. role and stage-2 action ----------------
        role_prop = random.bernoulli(k_role, 0.5, shape=(N,)).astype(jnp.float32)
        role_idx = jnp.where(role_prop == 1.0, 0, 1)  # 0=proposer 1=responder

        q_coop = Q_s2[idx, jnp.where(role_idx == 0, 0, 2)]
        q_def  = Q_s2[idx, jnp.where(role_idx == 0, 1, 3)]
        strategic_bonus = SM_b * 0.5 + SN_b * 0.3
        punish_insens   = SP_b * 0.3
        L2 = jnp.stack([q_coop, q_def + strategic_bonus], axis=-1) * beta_soft[:, None]
        L2 = L2 - punish_insens[:, None] * jnp.stack([jnp.abs(q_coop),
                                                       jnp.abs(q_def)], axis=-1)
        L2 = jnp.clip(L2, -30.0, 30.0)
        a2 = random.categorical(k_a2, L2, axis=-1)   # 0=coop/accept 1=def/reject

        # ---------------- 5. offer (proposer) ----------------
        # Full Fehr-Schmidt argmax over 11 discrete offers, scaled by tau_sens
        U_P = (E - offers[None, :]) \
              - alpha_eff[:, None] * relu_p_off[None, :] \
              - beta_guilt[:, None] * relu_n_off[None, :] \
              - s_spite[:, None] * relu_p_off[None, :] * 0.5
        U_P_scaled = U_P * (0.5 + 3.0 * tau_sens)[:, None]
        my_offer = offers[jnp.argmax(U_P_scaled, axis=-1)]        # (N,)
        received_offer = my_offer[chosen]                          # (N,)

        # ---------------- 6. responder-side utilities ----------------
        # My utility if I receive received_offer (I am responder)
        diff_me   = 2.0 * received_offer - E
        relu_me_p = jnp.maximum(diff_me, 0.0)
        relu_me_n = jnp.maximum(-diff_me, 0.0)
        my_U_R = received_offer \
                 - alpha_envy * relu_me_n \
                 - beta_guilt * relu_me_p \
                 + lam_norm
        my_accept = ((my_U_R >= 0.0) & is_responder & interacting).astype(jnp.float32)

        # Partner's utility if they receive my_offer (I am proposer)
        diff_p   = 2.0 * my_offer - E
        relu_p_p = jnp.maximum(diff_p, 0.0)
        relu_p_n = jnp.maximum(-diff_p, 0.0)
        partner_U_R = my_offer \
                      - alpha_envy[chosen] * relu_p_n \
                      - beta_guilt[chosen] * relu_p_p
        partner_accept = ((partner_U_R >= 0.0) & is_proposer & interacting).astype(jnp.float32)

        # ---------------- 7. reward ----------------
        r_P = jnp.where(partner_accept >= 1.0, E - my_offer, 0.0)
        r_R = jnp.where(my_accept >= 1.0, received_offer, 0.0)
        raw_reward = r_P + r_R
        # mood modulates reward from actual interactions only; no idle drain.
        # If raw_reward == 0 (idle / non-interacting), mood has zero effect.
        reward = jnp.clip(
            raw_reward * omega_soc * (1.0 + sal * 0.2 + 0.05 * mood),
            -50.0, 50.0)

        # ---------------- 8. Q-learning updates ----------------
        Q_mf_up = Q_mf[idx, a1] + lr_eff * (reward - Q_mf[idx, a1])
        Q_mf_new = Q_mf.at[idx, a1].set(jnp.clip(Q_mf_up, -100.0, 100.0))
        s2_idx = jnp.where(role_idx == 0, a2, 2 + a2)
        Q_s2_up = Q_s2[idx, s2_idx] + lr_eff * (reward - Q_s2[idx, s2_idx])
        Q_s2_new = Q_s2.at[idx, s2_idx].set(jnp.clip(Q_s2_up, -100.0, 100.0))

        # ---------------- 9. P_role learning ----------------
        role_onehot = jnp.stack([1.0 - role_prop, role_prop], axis=-1)
        P_role_new = P_role + eta_vol[:, None] * (role_onehot - P_role)
        P_role_new = P_role_new / jnp.maximum(P_role_new.sum(axis=-1, keepdims=True), 1e-6)

        # ---------------- 10. distress update ----------------
        rejected = is_responder & interacting & (my_U_R < 0.0)
        new_distress = (DISTRESS_DECAY * distress
                        + DISTRESS_PULL * baseline_dist
                        + (1.0 + N_sens) * rejected.astype(jnp.float32) * 0.05)

        # ---------------- 11. wealth and aggregates ----------------
        # Floor at 0: agents cannot hold negative resources in this model.
        # [SYNTH-PRIOR-COEFF] — spec does not specify borrowing; explicit choice.
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
               gini, jnp.mean(jnp.abs(Q_net_engage)), jnp.mean(reward))
        return (Q_mf_new, Q_s2_new, P_role_new, new_wealth, new_distress, a1), out
    return step

def run(params_pkl, n_ticks, seed=42):
    P_np = load_all(params_pkl)
    P = {k: jnp.asarray(v) for k, v in P_np.items()}
    nb, degs = build_ws()
    print(f"[ws] degree min={degs.min()} max={degs.max()} mean={degs.mean():.3f}")
    neighbors = jnp.asarray(nb, jnp.int32)

    Q_mf0   = jnp.zeros((WS_N, 2), jnp.float32)
    Q_s2_0  = jnp.zeros((WS_N, 4), jnp.float32)
    P_role0 = jnp.full((WS_N, 2), 0.5, jnp.float32)
    wealth0 = jnp.full((WS_N,), WEALTH_INIT, jnp.float32)
    baseline = (P['general_distress'] + 0.3*P['DASS_Dep']
                + 0.3*P['DASS_Anx'] + 0.3*P['DASS_Str']) / 1.9
    last0 = jnp.zeros((WS_N,), jnp.int32)

    step_fn = make_step(neighbors, P)
    keys = random.split(random.PRNGKey(seed), n_ticks)

    @jax.jit
    def run_inner(carry, ks):
        return lax.scan(step_fn, carry, ks)

    carry0 = (Q_mf0, Q_s2_0, P_role0, wealth0, baseline, last0)
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
            "P_role": _n(final[2]), "wealth": _n(final[3]),
            "distress": _n(final[4])}
