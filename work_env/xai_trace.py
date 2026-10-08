# xai_trace.py — Phase 3: XAI trace API.
#
# Re-runs the sim from a checkpoint to a target tick with the step function
# instrumented to record one designated agent's full state at every tick.
# Records both self-view and partner-view fields.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import hashlib
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax
import engine_v5 as E

TRACE_KEYS = [
    "Q_mf_a0", "Q_mf_a1",
    "Q_MB_a0", "Q_MB_a1",
    "Q_net_a0", "Q_net_a1",
    "w_mb_eff",
    "distress_pre",
    "a1", "s2", "a2",
    "P_hat_a1_0", "P_hat_a1_1",
    "P_hat_alt_0", "P_hat_alt_1",
    "reward",
    "interacting", "rejected",
    "partner_id",
    "partner_alpha_envy", "partner_beta_guilt",
    "partner_a2",
    "my_offer", "partner_accept",
]
TRACE_DIM = len(TRACE_KEYS)

def state_hash(d):
    h = hashlib.sha256()
    for k in ["Q_mf", "Q_s2", "P_hat", "wealth", "distress"]:
        h.update(np.ascontiguousarray(d[k], np.float32).tobytes())
    return h.hexdigest()

def _make_traced_step(neighbors, P, agent_i):
    N, K = neighbors.shape
    idx = jnp.arange(N)
    E_ = E.E_ENDOWMENT
    valid_mask = (neighbors >= 0)
    logits_pick = jnp.where(valid_mask, 0.0, -1e30)

    tau        = E.TAU_BASE * (0.5 + P['O'])
    alpha_envy = P['D'] + P['lambda_ineq']
    beta_guilt = P['A']
    s_spite    = P['s_spite']
    lr         = P['alpha_R']
    k_DD_true  = jnp.exp(jnp.clip(P['k_DD'], -10.0, 5.0))
    gamma      = 1.0 / (1.0 + k_DD_true)
    beta_soft  = P['beta_choice']
    phi        = P['phi_per']
    w_mb_base  = P['w_MB']
    C_b        = P['C']
    eta_vol    = P['eta_vol']
    N_sens     = P['N']
    E_bias     = P['E'] * E.E_ENGAGE
    a_inhib    = P['a_inhib'] * E.A_INHIB
    theta_T    = P['theta_T'] * E.T_THREAT
    u_U        = P['u_U'] * E.U_UNCERT
    SM_b       = P['S_M'] * E.SM_DEFECT
    SP_b       = P['S_P'] * E.SP_PUNISH
    SN_b       = P['S_N'] * E.SN_STATUS
    omega_soc  = P['omega_social']
    lam_norm   = P['lambda_norm'] * E.LAMBDA_NORM_A
    sal        = P['gamma_sal'] * E.SAL_GAIN
    mood       = P['m_t']
    baseline_dist = (P['general_distress']
                     + 0.3*P['DASS_Dep'] + 0.3*P['DASS_Anx'] + 0.3*P['DASS_Str']) / 1.9
    ti = int(agent_i)

    def step(carry, tick_key):
        Q_mf, Q_s2, P_hat, wealth, distress, last_a1 = carry
        (k_pick, k_coin, k_a1, k_a2, k_trust, k_role, _) = random.split(tick_key, 7)

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

        w_mb_eff = jnp.clip(
            w_mb_base * (1.0 + C_b * E.C_BOOST) * (1.0 - distress * E.KAPPA_DECAY),
            0.0, 1.0)
        best_s2 = jnp.max(Q_s2, axis=-1)
        Q_MB = jnp.sum(P_hat * best_s2[:, None, :], axis=-1)
        Q_net = w_mb_eff[:, None] * Q_MB + (1.0 - w_mb_eff[:, None]) * Q_mf

        engage_adv = E_bias - a_inhib
        avoid_adv  = theta_T + u_U
        persev = phi * jnp.where(last_a1 == 0, 1.0, 0.0)
        L1 = tau[:, None] * jnp.stack(
            [Q_net[:, 0] + engage_adv + persev, Q_net[:, 1] + avoid_adv], axis=-1)
        L1 = jnp.clip(L1, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a1 = random.categorical(k_a1, L1, axis=-1)

        common = random.bernoulli(k_role, E.P_COMMON, shape=(N,))
        s2 = jnp.where(common, a1, 1 - a1)

        q_s2_actions = Q_s2[idx, s2, :]
        q_s2_actions = jnp.where(
            q_s2_actions < 0.0,
            q_s2_actions * (1.0 - SP_b[:, None] * 0.5),
            q_s2_actions)
        dark_bias = -SM_b - SN_b
        action_bonus = jnp.stack(
            [dark_bias, dark_bias * 0.5, jnp.zeros_like(dark_bias), jnp.zeros_like(dark_bias)],
            axis=-1)
        L2 = beta_soft[:, None] * (q_s2_actions + action_bonus)
        L2 = jnp.clip(L2, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a2 = random.categorical(k_a2, L2, axis=-1)

        my_offer = E.ULTI_OFFERS[a2]
        received_offer = my_offer[chosen]
        d_me = 2.0 * received_offer - E_
        rp_me = jnp.maximum(d_me, 0.0)
        rn_me = jnp.maximum(-d_me, 0.0)
        my_U_R = received_offer - alpha_envy * rn_me - beta_guilt * rp_me + lam_norm
        my_accept = ((my_U_R >= 0.0) & is_responder).astype(jnp.float32)

        d_p = 2.0 * my_offer - E_
        rp_p = jnp.maximum(d_p, 0.0)
        rn_p = jnp.maximum(-d_p, 0.0)
        partner_U_R = my_offer - alpha_envy[chosen] * rn_p - beta_guilt[chosen] * rp_p
        partner_accept = ((partner_U_R >= 0.0) & is_proposer).astype(jnp.float32)

        r_ulti_P = jnp.where(partner_accept >= 1.0, E_ - my_offer, 0.0)
        r_ulti_R = jnp.where(my_accept >= 1.0, received_offer, 0.0)

        is_trust_sender  = is_responder
        is_trust_trustee = is_proposer
        my_send = E.TRUST_SENDS[a2]
        received_send = my_send[chosen]
        my_return_mult = E.TRUST_RETURN_MULT[a2]
        received_return_mult = my_return_mult[chosen]
        r_trust_sender = jnp.where(
            is_trust_sender,
            -my_send + E.TRUST_K * received_return_mult * my_send, 0.0)
        r_trust_trustee = jnp.where(
            is_trust_trustee,
            E.TRUST_K * received_send * (1.0 - my_return_mult / E.TRUST_K), 0.0)

        raw_reward = r_ulti_P + r_ulti_R + r_trust_sender + r_trust_trustee
        reward = jnp.clip(
            raw_reward * omega_soc * (1.0 + sal + 0.05 * mood),
            -E.REWARD_CLIP, E.REWARD_CLIP)
        interacting = (a1 == 0) & (a1[chosen] == 0) & is_mutual
        reward = jnp.where(interacting, reward, 0.0)

        update_mask = interacting.astype(jnp.float32)
        Q_s2_prev = Q_s2[idx, s2, a2]
        Q_s2_new_ = jnp.clip(
            Q_s2_prev + update_mask * lr * (reward - Q_s2_prev),
            -E.GRAD_CLIP, E.GRAD_CLIP)
        Q_s2_new = Q_s2.at[idx, s2, a2].set(Q_s2_new_)

        Q_mf_prev = Q_mf[idx, a1]
        v_next = jnp.max(Q_mf, axis=-1)
        Q_mf_new_ = jnp.clip(
            Q_mf_prev + update_mask * lr * (reward + gamma * v_next - Q_mf_prev),
            -E.GRAD_CLIP, E.GRAD_CLIP)
        Q_mf_new = Q_mf.at[idx, a1].set(Q_mf_new_)

        one_hot = jnp.stack([1.0 - s2.astype(jnp.float32),
                             s2.astype(jnp.float32)], axis=-1)
        P_at = P_hat[idx, a1, :]
        P_upd = P_at + update_mask[:, None] * eta_vol[:, None] * (one_hot - P_at)
        P_upd = jnp.clip(P_upd, 1e-4, 1.0)
        P_upd = P_upd / jnp.sum(P_upd, axis=-1, keepdims=True)
        P_hat_new = P_hat.at[idx, a1, :].set(P_upd)

        rejected = is_responder & interacting & (my_U_R < 0.0)
        new_distress = (E.DISTRESS_DECAY * distress
                        + E.DISTRESS_PULL * baseline_dist
                        + (1.0 + N_sens * E.N_DISTRESS) * rejected.astype(jnp.float32) * 0.05)
        new_wealth = jnp.maximum(wealth + reward, 0.0)

        n_pairs = jnp.sum(is_mutual.astype(jnp.float32)) / 2.0
        n_resp  = jnp.sum((is_responder & interacting).astype(jnp.float32))
        n_acc   = jnp.sum(my_accept * interacting)
        coop    = jnp.where(n_resp > 0, n_acc / n_resp, 0.0)
        ws_s = jnp.sort(new_wealth)
        tot  = jnp.sum(new_wealth)
        i1   = jnp.arange(1, N + 1, dtype=jnp.float32)
        gini = jnp.where(tot > 0,
                         (2.0 * jnp.sum(i1 * ws_s)) / (N * tot) - (N + 1.0) / N, 0.0)
        mean_a2 = jnp.mean(a2.astype(jnp.float32))

        partner_i = chosen[ti]
        trace = jnp.stack([
            Q_mf[ti, 0], Q_mf[ti, 1],
            Q_MB[ti, 0], Q_MB[ti, 1],
            Q_net[ti, 0], Q_net[ti, 1],
            w_mb_eff[ti],
            distress[ti],
            a1[ti].astype(jnp.float32),
            s2[ti].astype(jnp.float32),
            a2[ti].astype(jnp.float32),
            P_hat[ti, a1[ti], 0], P_hat[ti, a1[ti], 1],
            P_hat[ti, 1 - a1[ti], 0], P_hat[ti, 1 - a1[ti], 1],
            reward[ti],
            interacting[ti].astype(jnp.float32),
            rejected[ti].astype(jnp.float32),
            partner_i.astype(jnp.float32),
            alpha_envy[partner_i],
            beta_guilt[partner_i],
            a2[partner_i].astype(jnp.float32),
            my_offer[ti],
            partner_accept[ti],
        ])

        aggregates = (coop, n_pairs, jnp.mean(new_distress),
                      jnp.mean(w_mb_eff), jnp.sum(rejected.astype(jnp.float32)),
                      gini, jnp.mean(jnp.abs(Q_net)), jnp.mean(reward), mean_a2)
        return (Q_mf_new, Q_s2_new, P_hat_new, new_wealth, new_distress, a1), (aggregates, trace)
    return step

def trace_agent(ckpt_path, target_tick, agent_i, params_pkl, N=None):
    z = np.load(ckpt_path, allow_pickle=False)
    start_tick = int(z["tick"])
    seed = int(z["seed"])
    n_ticks = int(z["n_ticks"])
    N_ckpt = int(z["N"])
    if N is None:
        N = N_ckpt
    if N != N_ckpt:
        raise ValueError(f"N mismatch: ckpt {N_ckpt}, requested {N}")
    if not (0 <= agent_i < N):
        raise ValueError(f"agent_i {agent_i} out of range [0,{N})")
    if target_tick < start_tick:
        raise ValueError(f"target {target_tick} < ckpt tick {start_tick}")
    if target_tick > n_ticks:
        raise ValueError(f"target {target_tick} > n_ticks {n_ticks}")

    P_np = E.load_all(params_pkl)
    P = {k: jnp.asarray(v[:N]) for k, v in P_np.items()}
    nb, _ = E.build_ws(n=N, k=6, p=0.1, seed=0)
    neighbors = jnp.asarray(nb, jnp.int32)

    carry = (
        jnp.asarray(z["Q_mf"]), jnp.asarray(z["Q_s2"]),
        jnp.asarray(z["P_hat"]), jnp.asarray(z["wealth"]),
        jnp.asarray(z["distress"]), jnp.asarray(z["last_a1"]),
    )
    step_fn = _make_traced_step(neighbors, P, agent_i)
    keys = random.split(random.PRNGKey(seed), n_ticks)
    remaining = keys[start_tick:target_tick]

    @jax.jit
    def run_inner(carry, ks):
        return lax.scan(step_fn, carry, ks)

    final_carry, (aggregates, traces) = run_inner(carry, remaining)
    jax.block_until_ready(final_carry)
    Q_mf_f, Q_s2_f, P_hat_f, w_f, d_f, _last = final_carry
    final_dict = {
        "Q_mf": np.asarray(Q_mf_f), "Q_s2": np.asarray(Q_s2_f),
        "P_hat": np.asarray(P_hat_f), "wealth": np.asarray(w_f),
        "distress": np.asarray(d_f),
    }
    return {
        "start_tick": start_tick, "target_tick": target_tick, "agent_i": agent_i,
        "N": N, "keys": TRACE_KEYS,
        "values": np.asarray(traces),
        "aggregates": np.asarray(aggregates),
        "final_state_hash": state_hash(final_dict),
        "provenance": E.PARAM_PROVENANCE,
        "synthetic_priors_in_effect": E.SYNTHETIC_PRIOR_PARAMS,
        "active_flags": [
            "SYNTHETIC_PRIOR_PARAMS=" + ",".join(E.SYNTHETIC_PRIOR_PARAMS),
            "COPULA-COLUMN-IS-LOG-K-DD",
            "MF-GENERALIZED",
            "REWARD-SCALE-ASYMMETRY",
        ],
    }
