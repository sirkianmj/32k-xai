# p6c_v5.py — Phase 6c v5: regularized + cross-validated calibration.
#
# Changes over v4:
#   1. L2 penalty on calibration scalars. Prevents degenerate corner solutions
#      (e.g. BETA_OFF -> 0.001 collapses offer distribution to one value).
#   2. Tighter bounds on semantic scalars (physical plausibility).
#   3. Multi-seed averaging: each theta evaluated on 3 seeds, loss = mean + std.
#   4. Train / holdout split on benchmarks:
#        train targets   = (offer, send, return)
#        holdout targets = (rejection, gini)
#      Report BOTH. A model that only fits train is overfit.
#
# Scientific criterion for success:
#   train loss   in [0.5, 5.0]  (nonzero, low)
#   holdout loss in [0.0, 6.0]  (nonzero, low)
#   both parameterized by physically plausible theta.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import time, json
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from jax import random, lax
from scipy.optimize import differential_evolution
import engine_v5 as E

OUT = Path(__file__).parent
PARAMS = Path(os.environ.get(
    "P6C_PARAMS",
    str(Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"),
))
SEED = 42

CALIB_NAMES = ["TAU_BASE", "BETA_OFF", "BETA_SEND", "BETA_ACC",
               "BETA_RET", "BIAS_OFF", "BIAS_SEND", "BIAS_RET", "LAMBDA_FS"]
# Tighter bounds — physical plausibility
BOUNDS = [(0.5, 2.5),   # TAU_BASE     : softmax temperature (was 0.3-4.0)
          (0.5, 3.0),   # BETA_OFF     : offer sharpness  (was 0.3-6.0)
          (0.5, 3.0),   # BETA_SEND    : send sharpness
          (0.5, 4.0),   # BETA_ACC     : accept sharpness
          (0.5, 3.0),   # BETA_RET     : return sharpness
          (-1.5, 1.5),  # BIAS_OFF     : offer bias       (was -3..3)
          (-1.5, 1.5),  # BIAS_SEND    : send bias
          (-1.5, 1.5),  # BIAS_RET     : return bias
          (0.5, 6.0)]   # LAMBDA_FS    : FS accept weight (was 0..8)

# Train and holdout split
TARGETS_ALL = np.array([0.40, 0.16, 0.502, 0.372, 0.40])
SE_ALL      = np.array([0.05, 0.05, 0.130, 0.120, 0.12])
TRAIN_IDX   = [0, 2, 3]      # offer, send, return
HOLDOUT_IDX = [1, 4]         # rejection, gini

TARGETS_TRAIN = TARGETS_ALL[TRAIN_IDX]
SE_TRAIN      = SE_ALL[TRAIN_IDX]
TARGETS_HOLD  = TARGETS_ALL[HOLDOUT_IDX]
SE_HOLD       = SE_ALL[HOLDOUT_IDX]

L2_LAMBDA     = 0.02   # regularization strength
SEED_COUNT    = 3      # multi-seed averaging

_SRC = E.load_all(PARAMS, required=False)
_N_ALL = len(_SRC["O"])

def build_simulator(N, T, trace_agent_i=None, shocks=None, ws_k=6, ws_p=0.1, return_diag=False):
    idx = np.random.default_rng(SEED).integers(0, _N_ALL, size=N)
    P = {k: jnp.asarray(v[idx]) for k, v in _SRC.items()}

    CB       = P["C"] * E.C_BOOST
    eta_vol  = P["eta_vol"]
    E_bias   = P["E"] * E.E_ENGAGE
    a_inhib  = P["a_inhib"] * E.A_INHIB
    theta_T  = P["theta_T"] * E.T_THREAT
    u_U      = P["u_U"] * E.U_UNCERT
    omega    = P["omega_social"]
    sal      = P["gamma_sal"] * E.SAL_GAIN
    mood     = P["m_t"]
    N_sens   = P["N"]
    lr       = P["alpha_R"]
    phi      = P["phi_per"]
    w_base   = P["w_MB"]
    O_p      = P["O"]
    gamma    = 1.0/(1.0 + jnp.exp(jnp.clip(P["k_DD"], -10.0, 5.0)))
    baseline = (P["general_distress"] + 0.3*P["DASS_Dep"]
                + 0.3*P["DASS_Anx"] + 0.3*P["DASS_Str"]) / 1.9

    nb, _ = E.build_ws(n=N, k=int(ws_k), p=float(ws_p), seed=0)
    NB = jnp.asarray(nb, jnp.int32)
    AR = jnp.arange(N)
    LOGITS_PICK = jnp.where(NB >= 0, 0.0, -1e30)
    k_default_np = (nb >= 0).sum(axis=1).astype(np.int32)

    OFFER_VALS = jnp.array([2.0, 4.0, 5.0, 7.0])
    SEND_VALS  = jnp.array([0.0, 3.0, 6.0, 9.0])
    RET_VALS   = jnp.array([0.0, 1.0, 2.0, 3.0])

    Q_MF0   = jnp.zeros((N, 2), jnp.float32)
    Q_OFF0  = jnp.zeros((N, 2, 4), jnp.float32)
    Q_SEND0 = jnp.zeros((N, 2, 4), jnp.float32)
    Q_ACC0  = jnp.zeros((N, 2, 2), jnp.float32)
    Q_RET0  = jnp.zeros((N, 2, 4), jnp.float32)
    P_HAT0  = jnp.zeros((N, 2, 2), jnp.float32)
    P_HAT0  = P_HAT0.at[:, 0, 0].set(E.P_COMMON).at[:, 0, 1].set(1.0 - E.P_COMMON)
    P_HAT0  = P_HAT0.at[:, 1, 0].set(1.0 - E.P_COMMON).at[:, 1, 1].set(E.P_COMMON)
    W0      = jnp.full((N,), E.WEALTH_INIT, jnp.float32)
    LAST0   = jnp.full((N,), -1, jnp.int32)

    # ---- Phase 4: shock precomputation ----
    # shocks: list of dicts, each with keys tick, duration, type, magnitude,
    # target_frac (None or float in (0,1]), target_seed (int).
    #   resource_scarcity: multiplicative reduction on the reward modifier
    #   distress_injection: additive bump on the per-agent baseline distress
    if shocks:
        resource_mod_np = np.ones((T, N), np.float32)
        baseline_mod_np = np.zeros((T, N), np.float32)
        keff_np = np.tile(k_default_np[None, :], (T, 1)).astype(np.int32)
        redist_np = np.zeros((T, N), np.float32)
        for s in shocks:
            t0 = int(s["tick"]); dur = int(s["duration"])
            mag = float(s["magnitude"]); frac = s.get("target_frac")
            ts = int(s.get("target_seed", 0))
            if "target_mask" in s and s["target_mask"] is not None:
                mask = np.asarray(s["target_mask"], dtype=bool)
                if mask.shape != (N,):
                    raise ValueError(
                        f"target_mask length {mask.shape} does not match N={N}")
            elif frac is None:
                mask = np.ones(N, dtype=bool)
            else:
                rng = np.random.default_rng(ts)
                mask = rng.random(N) < float(frac)
            t1 = min(T, t0 + dur)
            stype = s["type"]
            if stype == "resource_scarcity":
                resource_mod_np[t0:t1, mask] *= (1.0 - mag)
            elif stype == "distress_injection":
                baseline_mod_np[t0:t1, mask] += mag
            elif stype == "network_thinning":
                reduce = np.maximum(np.round(mag * k_default_np[mask]).astype(np.int32), 0)
                for t in range(t0, t1):
                    keff_np[t, mask] = np.maximum(keff_np[t, mask] - reduce, 1)
            elif stype == "wealth_redistribution":
                redist_np[t0:t1, mask] += np.clip(mag, 0.0, 1.0)
            else:
                raise ValueError(f"unknown shock type: {stype}")
        resource_mod_jax = jnp.asarray(resource_mod_np)
        baseline_mod_jax = jnp.asarray(baseline_mod_np)
        keff_jax = jnp.asarray(keff_np, jnp.int32)
        redist_jax = jnp.asarray(redist_np)
    else:
        resource_mod_jax = jnp.ones((T, N), jnp.float32)
        baseline_mod_jax = jnp.zeros((T, N), jnp.float32)
        keff_jax = jnp.tile(jnp.asarray(k_default_np, jnp.int32)[None, :], (T, 1))
        redist_jax = jnp.zeros((T, N), jnp.float32)

    def step(carry, scan_in):
        tk, rm_t, bm_t, keff_t, redist_t = scan_in
        calib, Q_mf, Q_off, Q_send, Q_acc, Q_ret, P_hat, wealth, dist, last_a1 = carry
        TAU_B, B_OFF, B_SEND, B_ACC, B_RET, BI_OFF, BI_SEND, BI_RET, LAMB_FS = (
            calib[0], calib[1], calib[2], calib[3],
            calib[4], calib[5], calib[6], calib[7], calib[8])
        kp, kc, ka1, kd, ke, kf, kg, kh, _ = random.split(tk, 9)

        _arange_k = jnp.arange(NB.shape[1])
        LOGITS_PICK_t = jnp.where(
            (NB >= 0) & (_arange_k[None, :] < keff_t[:, None]),
            0.0, -1e30)
        k_idx = random.categorical(kp, LOGITS_PICK_t, axis=-1)
        ch = NB[AR, k_idx]; mut = (ch[ch] == AR)
        ca = random.bernoulli(kc, 0.5, shape=(N,)).astype(jnp.float32)
        cp = ca[jnp.minimum(AR, ch)]
        il = AR < ch
        pp = jnp.where(cp == 1.0, il, ~il)
        isP = pp & mut; isR = (~pp) & mut

        tau = TAU_B * (0.5 + O_p)
        w_eff = jnp.clip(w_base * (1.0 + CB) * (1.0 - dist * E.KAPPA_DECAY), 0.0, 1.0)
        best_off = jnp.max(Q_off, axis=-1)
        best_send = jnp.max(Q_send, axis=-1)
        Q_MB = 0.5 * jnp.sum(P_hat * best_off[:, None, :], axis=-1) \
             + 0.5 * jnp.sum(P_hat * best_send[:, None, :], axis=-1)
        Q_net = w_eff[:, None] * Q_MB + (1.0 - w_eff[:, None]) * Q_mf
        eng = E_bias - a_inhib; avd = theta_T + u_U
        per = phi * jnp.where(last_a1 == 0, 1.0, 0.0)
        L1 = jnp.clip(tau[:, None] * jnp.stack(
            [Q_net[:, 0] + eng + per, Q_net[:, 1] + avd], axis=-1),
            -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a1 = random.categorical(ka1, L1, axis=-1)
        s2 = jnp.where(random.bernoulli(kh, E.P_COMMON, shape=(N,)), a1, 1 - a1)

        alpha_e = P["D"] + P["lambda_ineq"]
        beta_g  = P["A"]

        bias_off = jnp.stack([BI_OFF, -BI_OFF/2, jnp.zeros_like(BI_OFF), -BI_OFF/2], axis=-1)
        L_off = jnp.clip(B_OFF * (Q_off[AR, s2, :] + bias_off), -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a_off = random.categorical(kd, L_off, axis=-1)

        bias_send = jnp.stack([-BI_SEND, jnp.zeros_like(BI_SEND), jnp.zeros_like(BI_SEND), BI_SEND], axis=-1)
        L_send = jnp.clip(B_SEND * (Q_send[AR, s2, :] + bias_send), -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a_send = random.categorical(ke, L_send, axis=-1)

        my_off = OFFER_VALS[a_off]
        my_send = SEND_VALS[a_send]
        rec_off = my_off[ch]

        # Canonical Fehr-Schmidt (1999, QJE, eq. 1):
        #   U = my_payoff - alpha * max(other - mine, 0) - beta * max(mine - other, 0)
        # As responder: my_payoff = rec_off, other_payoff = E - rec_off.
        my_payoff    = rec_off
        other_payoff = E.E_ENDOWMENT - rec_off
        diff = other_payoff - my_payoff
        inequity_neg = jnp.maximum( diff, 0.0)   # other better off
        inequity_pos = jnp.maximum(-diff, 0.0)   # I better off
        U_accept = my_payoff - alpha_e * inequity_neg - beta_g * inequity_pos

        L_acc_accept = B_ACC * Q_acc[AR, s2, 0] + LAMB_FS * U_accept
        L_acc_reject = B_ACC * Q_acc[AR, s2, 1]
        L_acc = jnp.clip(jnp.stack([L_acc_accept, L_acc_reject], axis=-1),
                         -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a_acc = random.categorical(kf, L_acc, axis=-1)

        bias_ret = jnp.stack([-BI_RET, jnp.zeros_like(BI_RET), jnp.zeros_like(BI_RET), BI_RET], axis=-1)
        L_ret = jnp.clip(B_RET * (Q_ret[AR, s2, :] + bias_ret), -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a_ret = random.categorical(kg, L_ret, axis=-1)

        my_rm = RET_VALS[a_ret]
        rec_send = my_send[ch]
        rec_rm = my_rm[ch]
        my_reject = a_acc
        partner_reject = a_acc[ch]

        partner_accepts_me = (partner_reject == 0) & isP
        my_acceptance = ((my_reject == 0) & isR).astype(jnp.float32)

        r_off = jnp.where(partner_accepts_me, E.E_ENDOWMENT - my_off, 0.0)
        r_acc = jnp.where(my_acceptance >= 1.0, rec_off, 0.0)
        r_send = jnp.where(isR, -my_send + E.TRUST_K * rec_rm * my_send, 0.0)
        r_ret = jnp.where(isP, E.TRUST_K * rec_send * (1.0 - my_rm / E.TRUST_K), 0.0)

        mult = omega * (1.0 + sal + 0.05 * mood) * rm_t
        r_off_m = jnp.clip(r_off * mult, -E.REWARD_CLIP, E.REWARD_CLIP)
        r_acc_m = jnp.clip(r_acc * mult, -E.REWARD_CLIP, E.REWARD_CLIP)
        r_send_m = jnp.clip(r_send * mult, -E.REWARD_CLIP, E.REWARD_CLIP)
        r_ret_m = jnp.clip(r_ret * mult, -E.REWARD_CLIP, E.REWARD_CLIP)

        inter = (a1 == 0) & (a1[ch] == 0) & mut
        um = inter.astype(jnp.float32)

        q_off_v = Q_off[AR, s2, a_off]
        upd_off = jnp.where(isP, um, 0.0)
        Q_off_n = Q_off.at[AR, s2, a_off].set(jnp.clip(
            q_off_v + upd_off * lr * (r_off_m - q_off_v), -E.GRAD_CLIP, E.GRAD_CLIP))

        q_send_v = Q_send[AR, s2, a_send]
        upd_send = jnp.where(isR, um, 0.0)
        Q_send_n = Q_send.at[AR, s2, a_send].set(jnp.clip(
            q_send_v + upd_send * lr * (r_send_m - q_send_v), -E.GRAD_CLIP, E.GRAD_CLIP))

        q_acc_v = Q_acc[AR, s2, a_acc]
        upd_acc = jnp.where(isR, um, 0.0)
        Q_acc_n = Q_acc.at[AR, s2, a_acc].set(jnp.clip(
            q_acc_v + upd_acc * lr * (r_acc_m - q_acc_v), -E.GRAD_CLIP, E.GRAD_CLIP))

        q_ret_v = Q_ret[AR, s2, a_ret]
        upd_ret = jnp.where(isP, um, 0.0)
        Q_ret_n = Q_ret.at[AR, s2, a_ret].set(jnp.clip(
            q_ret_v + upd_ret * lr * (r_ret_m - q_ret_v), -E.GRAD_CLIP, E.GRAD_CLIP))

        total_r = jnp.clip((r_off + r_acc + r_send + r_ret) * mult,
                           -E.REWARD_CLIP, E.REWARD_CLIP)
        total_r = jnp.where(inter, total_r, 0.0)
        qmv = Q_mf[AR, a1]; vn = jnp.max(Q_mf, axis=-1)
        Q_mfn = Q_mf.at[AR, a1].set(jnp.clip(
            qmv + um * lr * (total_r + gamma * vn - qmv), -E.GRAD_CLIP, E.GRAD_CLIP))

        oh = jnp.stack([1.0-s2.astype(jnp.float32), s2.astype(jnp.float32)], axis=-1)
        Pat = P_hat[AR, a1, :]
        Pu = Pat + um[:, None] * eta_vol[:, None] * (oh - Pat)
        Pu = jnp.clip(Pu, 1e-4, 1.0); Pu = Pu / jnp.sum(Pu, axis=-1, keepdims=True)
        P_hatn = P_hat.at[AR, a1, :].set(Pu)

        rej = isR & inter & (my_reject == 1)
        baseline_eff = jnp.clip(baseline + bm_t, 0.0, 5.0)
        dn = (E.DISTRESS_DECAY * dist + E.DISTRESS_PULL * baseline_eff
              + (1.0 + N_sens * E.N_DISTRESS) * rej.astype(jnp.float32) * 0.05)
        wn_raw = jnp.maximum(wealth + total_r, 0.0)
        mean_w = jnp.mean(wn_raw)
        wn = jnp.maximum(wn_raw + redist_t * (mean_w - wn_raw), 0.0)

        cap = jnp.stack([
            jnp.where(inter & isP, my_off, 0.0),
            jnp.where(inter & isR, (my_reject == 0).astype(jnp.float32), 0.0),
            jnp.where(inter & isR, 1.0, 0.0),
            jnp.where(inter & isP, 1.0, 0.0),
            jnp.where(inter & isR, my_send, 0.0),
            jnp.where(inter & isR, 1.0, 0.0),
            jnp.where(inter & isR, rec_rm * my_send, 0.0),
            jnp.where(inter & isP, rec_send, 0.0),
            jnp.where(inter & isP, my_rm * rec_send, 0.0),
            jnp.where(inter & isP, 1.0, 0.0),
            # Phase 5 step 3: per-tick state for time-series
            dn,                    # 10 — new distress after update
            wn,                    # 11 — new wealth after update
            w_eff,                 # 12 — arbitration weight used this tick
        ], axis=-1)

        if trace_agent_i is not None:
            ti = int(trace_agent_i)
            trace_vec = jnp.stack([
                Q_mf[ti, 0], Q_mf[ti, 1],
                Q_MB[ti, 0], Q_MB[ti, 1],
                Q_net[ti, 0], Q_net[ti, 1],
                w_eff[ti], tau[ti],
                eng[ti], avd[ti], per[ti],
                L1[ti, 0], L1[ti, 1],
                a1[ti].astype(jnp.float32), s2[ti].astype(jnp.float32),
                P_hat[ti, a1[ti], 0], P_hat[ti, a1[ti], 1],
                P_hat[ti, 1 - a1[ti], 0], P_hat[ti, 1 - a1[ti], 1],
                a_off[ti].astype(jnp.float32), my_off[ti],
                Q_off[ti, s2[ti], a_off[ti]],
                a_send[ti].astype(jnp.float32), my_send[ti],
                Q_send[ti, s2[ti], a_send[ti]],
                a_acc[ti].astype(jnp.float32),
                Q_acc[ti, s2[ti], 0], Q_acc[ti, s2[ti], 1],
                U_accept[ti], inequity_neg[ti], inequity_pos[ti],
                a_ret[ti].astype(jnp.float32), my_rm[ti],
                Q_ret[ti, s2[ti], a_ret[ti]],
                r_off[ti], r_acc[ti], r_send[ti], r_ret[ti], total_r[ti],
                ch[ti].astype(jnp.float32),
                alpha_e[ch[ti]], beta_g[ch[ti]],
                a_off[ch[ti]].astype(jnp.float32),
                a_send[ch[ti]].astype(jnp.float32),
                a_ret[ch[ti]].astype(jnp.float32),
                rec_off[ti], rec_send[ti], rec_rm[ti],
                partner_reject[ti].astype(jnp.float32),
                a1[ch[ti]].astype(jnp.float32),
                inter[ti].astype(jnp.float32),
                rej[ti].astype(jnp.float32),
                mut[ti].astype(jnp.float32),
                dist[ti], baseline_eff[ti], wealth[ti],
            ])
            return (calib, Q_mfn, Q_off_n, Q_send_n, Q_acc_n, Q_ret_n,
                    P_hatn, wn, dn, a1), (cap, trace_vec)
        return (calib, Q_mfn, Q_off_n, Q_send_n, Q_acc_n, Q_ret_n,
                P_hatn, wn, dn, a1), cap

    def _run_with_seed(calib_vec, seed_val):
        keys = random.split(random.PRNGKey(seed_val), T)
        carry0 = (calib_vec, Q_MF0, Q_OFF0, Q_SEND0, Q_ACC0, Q_RET0,
                  P_HAT0, W0, baseline, LAST0)
        final, _scan_out = lax.scan(step, carry0, (keys, resource_mod_jax, baseline_mod_jax, keff_jax, redist_jax))
        if trace_agent_i is not None:
            cap, traces = _scan_out
        else:
            cap = _scan_out
            traces = None

        wealth = final[7]
        c0=cap[:,:,0]; c1=cap[:,:,1]; c2=cap[:,:,2]
        c4=cap[:,:,4]; c5=cap[:,:,5]; c7=cap[:,:,7]; c8=cap[:,:,8]; c9=cap[:,:,9]

        om = c0.reshape(-1); omsk = om > 0.0
        n_off = jnp.maximum(jnp.sum(omsk), 1.0)
        ug_offer = jnp.where(jnp.sum(omsk)>0, jnp.sum(om*omsk)/n_off/E.E_ENDOWMENT, 0.0)

        rm = c2.reshape(-1) > 0.0
        n_resp = jnp.maximum(jnp.sum(rm), 1.0)
        ug_rej = jnp.where(jnp.sum(rm)>0, 1.0 - jnp.sum(c1.reshape(-1)*rm)/n_resp, 0.5)

        sm = c5.reshape(-1) > 0.0
        n_send = jnp.maximum(jnp.sum(sm), 1.0)
        tg_send = jnp.where(jnp.sum(sm)>0, jnp.sum(c4.reshape(-1)*sm)/n_send/E.E_ENDOWMENT, 0.0)

        tm = c9.reshape(-1) > 0.0
        recv = c7.reshape(-1); ret = c8.reshape(-1)
        valid = tm & (recv > 0.0)
        n_val = jnp.maximum(jnp.sum(valid), 1.0)
        ret_frac = jnp.where(jnp.sum(valid)>0,
                             jnp.sum(ret * valid / jnp.where(recv>0, recv, 1.0))/n_val/3.0, 0.0)

        ws = jnp.sort(wealth); n = wealth.shape[0]
        cum = jnp.cumsum(ws); tot = cum[-1]
        gini = jnp.where(tot>0, (n+1 - 2.0*jnp.sum(cum)/tot)/n, 0.5)

        metrics = jnp.stack([ug_offer, ug_rej, tg_send, ret_frac, gini])

        # Phase 5 step 3: time-series aggregates
        # cap shape: (T, N, 13). Column 10 = distress, 11 = wealth, 12 = w_eff.
        resp_per_tick = cap[:, :, 2].sum(axis=1)                      # (T,)
        acc_per_tick  = cap[:, :, 1].sum(axis=1)                      # (T,)
        coop_per_tick = jnp.where(resp_per_tick > 0,
                                  acc_per_tick / jnp.maximum(resp_per_tick, 1.0),
                                  jnp.nan)
        dist_per_tick = cap[:, :, 10].mean(axis=1)                    # (T,)
        wmb_per_tick  = cap[:, :, 12].mean(axis=1)                    # (T,)
        wealth_per_tick = cap[:, :, 11]                               # (T, N)
        n_agents = wealth_per_tick.shape[1]
        ws_sorted = jnp.sort(wealth_per_tick, axis=1)
        ii = jnp.arange(1, n_agents + 1, dtype=jnp.float32)
        cum = jnp.cumsum(ws_sorted, axis=1)
        tot = cum[:, -1:]
        gini_per_tick = jnp.where(
            tot[:, 0] > 0,
            (2.0 * jnp.sum(ii[None, :] * ws_sorted, axis=1)) / (n_agents * tot[:, 0])
              - (n_agents + 1.0) / n_agents,
            0.5)
        timeseries = jnp.stack([coop_per_tick, dist_per_tick,
                                wmb_per_tick, gini_per_tick], axis=0)  # (4, T)

        if trace_agent_i is not None:
            return metrics, timeseries, traces
        return metrics, timeseries

    def simulate(calib_vec, seed_val=SEED, return_timeseries=False):
        if trace_agent_i is not None:
            m, _ts, tr = _run_with_seed(calib_vec, seed_val)
            return m, tr
        m, ts = _run_with_seed(calib_vec, seed_val)
        if return_timeseries:
            return m, ts
        return m

    # Multi-seed averaged simulate (returns mean and std over SEED_COUNT seeds)
    @jax.jit
    def simulate_multiseed(calib_vec):
        seeds = jnp.array([SEED, SEED+1, SEED+2], dtype=jnp.int32)
        def one(seed_val):
            r = _run_with_seed(calib_vec, seed_val)
            return r[0]  # metrics is always first
        # vmap over the 3 seeds
        all_res = jax.vmap(one)(seeds)   # (3, 5)
        mean_res = all_res.mean(axis=0)
        std_res  = all_res.std(axis=0)
        return mean_res, std_res

    if return_diag:
        return simulate, simulate
    return simulate_multiseed, simulate

print("Building small simulator (8000 x 250) ...", flush=True)
t0 = time.time()
SIM_SMALL_MULTI, SIM_SMALL = build_simulator(8000, 250)
_ = SIM_SMALL_MULTI(jnp.array([1.0, 1.5, 1.5, 2.0, 1.5, 0.0, 0.0, 0.0, 2.0], jnp.float32))[0].block_until_ready()
print(f"  compile wall = {time.time()-t0:.1f}s", flush=True)

def loss(calib_vec):
    """Regularized train loss with multi-seed averaging."""
    c = jnp.asarray(calib_vec, jnp.float32)
    mean_res, std_res = SIM_SMALL_MULTI(c)

    # Train benchmarks only
    plat_train = mean_res[jnp.array(TRAIN_IDX)]
    train_loss = jnp.sum(((plat_train - TARGETS_TRAIN) / SE_TRAIN) ** 2)

    # Multi-seed stability penalty
    stability = jnp.sum((std_res / SE_ALL) ** 2)

    # L2 penalty on calibration scalars (keep them near midpoint of bounds)
    midpoints = jnp.array([(lo+hi)/2 for lo, hi in BOUNDS], jnp.float32)
    scales    = jnp.array([(hi-lo)/2 for lo, hi in BOUNDS], jnp.float32)
    l2 = jnp.sum(((c - midpoints) / scales) ** 2)

    total = train_loss + 0.5 * stability + L2_LAMBDA * l2
    return float(total)

def holdout_loss(calib_vec):
    """Evaluate holdout benchmarks."""
    c = jnp.asarray(calib_vec, jnp.float32)
    mean_res, _ = SIM_SMALL_MULTI(c)
    plat_hold = mean_res[jnp.array(HOLDOUT_IDX)]
    return float(jnp.sum(((plat_hold - TARGETS_HOLD) / SE_HOLD) ** 2))

if __name__ == "__main__":
    def log(m): print(m, flush=True)
    log("=" * 72)
    log("PHASE 6c v5 — regularized + cross-validated calibration")
    log("=" * 72)
    log(f"Device: {jax.devices()[0]}")
    log(f"Train targets (3):   offer={TARGETS_TRAIN[0]}, send={TARGETS_TRAIN[1]}, ret={TARGETS_TRAIN[2]}")
    log(f"Holdout targets (2): rej={TARGETS_HOLD[0]}, gini={TARGETS_HOLD[1]}")
    log(f"L2 lambda: {L2_LAMBDA}, SEED_COUNT: {SEED_COUNT}")

    init = np.array([1.0, 1.5, 1.5, 2.0, 1.5, 0.0, 0.0, 0.0, 2.0], dtype=np.float32)
    log(f"\nLoss at init: train={loss(init):.4f}  holdout={holdout_loss(init):.4f}")

    log("\nDifferential evolution (popsize=15, maxiter=25, no polish) ...")
    t0 = time.time()
    gen = [0]
    def cb(xk, convergence):
        gen[0] += 1
        if gen[0] % 5 == 0:
            log(f"  gen {gen[0]:>3}  train={float(convergence):.4f}  "
                f"holdout={holdout_loss(xk):.4f}  wall={time.time()-t0:.1f}s")
    result = differential_evolution(
        loss, BOUNDS, seed=SEED, maxiter=25, popsize=15,
        tol=1e-6, polish=False, workers=1, updating="deferred",
        init="sobol", callback=cb,
    )
    log(f"  DE wall = {time.time()-t0:.1f}s")
    log(f"  best train loss = {result.fun:.4f}")
    log(f"  holdout loss    = {holdout_loss(result.x):.4f}")
    log(f"  θ = {dict(zip(CALIB_NAMES, [round(float(v),4) for v in result.x]))}")

    with open(OUT / "p6c_results.json", "w") as f:
        json.dump({
            "calib": dict(zip(CALIB_NAMES, [float(v) for v in result.x])),
            "train_loss": float(result.fun),
            "holdout_loss": holdout_loss(result.x),
        }, f, indent=2)
    log(f"Saved: {OUT / 'p6c_results.json'}")
