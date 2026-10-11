# p6c_decoupled.py — Phase 6c: decoupled role-actions + joint calibration.
#
# Architectural fix: four role-specific action heads replace the single a2:
#   Q_offer  (N, 2, 4): proposer ultimatum offers {2,4,5,7}
#   Q_send   (N, 2, 4): trust sender sends {0,3,6,9}
#   Q_accept (N, 2, 2): responder accept (0) / reject (1)
#   Q_return (N, 2, 4): trustee return mult {0,1,2,3}
#
# Calibration: 6 scalars, differential evolution on the FULL simulator
# (32k x 500, ~0.13 s per eval). No small-sim / full-sim mismatch.
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
PARAMS = Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"
SEED = 42

CALIB_NAMES = ["TAU_BASE", "ALPHA_SCALE", "BETA_SCALE",
               "BETA_ACCEPT", "LAM_NORM_SCALE", "TRUSTEE_RETURN_OFFSET"]
BOUNDS = [(0.3, 4.0), (0.2, 4.0), (0.2, 4.0),
          (0.1, 8.0), (0.0, 3.0), (-0.5, 1.5)]

TARGETS = np.array([0.40, 0.16, 0.502, 0.372, 0.40])
SE      = np.array([0.05, 0.05, 0.130, 0.120, 0.12])

_SRC = E.load_all(PARAMS)
_N_ALL = len(_SRC["O"])

def build_simulator(N, T, seed):
    idx = np.random.default_rng(seed).integers(0, _N_ALL, size=N)
    P = {k: jnp.asarray(v[idx]) for k, v in _SRC.items()}

    CB       = P["C"] * E.C_BOOST
    eta_vol  = P["eta_vol"]
    E_bias   = P["E"] * E.E_ENGAGE
    a_inhib  = P["a_inhib"] * E.A_INHIB
    theta_T  = P["theta_T"] * E.T_THREAT
    u_U      = P["u_U"] * E.U_UNCERT
    SM_b     = P["S_M"] * E.SM_DEFECT
    SP_b     = P["S_P"] * E.SP_PUNISH
    SN_b     = P["S_N"] * E.SN_STATUS
    omega    = P["omega_social"]
    sal      = P["gamma_sal"] * E.SAL_GAIN
    mood     = P["m_t"]
    lambda_e = P["D"] + P["lambda_ineq"]
    A_e      = P["A"]
    lambda_n = P["lambda_norm"] * E.LAMBDA_NORM_A
    N_sens   = P["N"]
    lr       = P["alpha_R"]
    beta_s   = P["beta_choice"]
    phi      = P["phi_per"]
    w_base   = P["w_MB"]
    O_p      = P["O"]
    gamma    = 1.0/(1.0 + jnp.exp(jnp.clip(P["k_DD"], -10.0, 5.0)))
    baseline = (P["general_distress"] + 0.3*P["DASS_Dep"]
                + 0.3*P["DASS_Anx"] + 0.3*P["DASS_Str"]) / 1.9

    nb, _ = E.build_ws(n=N, k=6, p=0.1, seed=0)
    NB = jnp.asarray(nb, jnp.int32)
    AR = jnp.arange(N)
    LOGITS_PICK = jnp.where(NB >= 0, 0.0, -1e30)

    Q_MF0   = jnp.zeros((N, 2), jnp.float32)
    Q_OFFER0 = jnp.zeros((N, 2, 4), jnp.float32)
    Q_SEND0  = jnp.zeros((N, 2, 4), jnp.float32)
    Q_ACC0   = jnp.zeros((N, 2, 2), jnp.float32)
    Q_RET0   = jnp.zeros((N, 2, 4), jnp.float32)
    P_HAT0  = jnp.zeros((N, 2, 2), jnp.float32)
    P_HAT0  = P_HAT0.at[:, 0, 0].set(E.P_COMMON).at[:, 0, 1].set(1.0 - E.P_COMMON)
    P_HAT0  = P_HAT0.at[:, 1, 0].set(1.0 - E.P_COMMON).at[:, 1, 1].set(E.P_COMMON)
    W0      = jnp.full((N,), E.WEALTH_INIT, jnp.float32)
    LAST0   = jnp.full((N,), -1, jnp.int32)

    def step(carry, tk):
        calib, Q_mf, Q_off, Q_send, Q_acc, Q_ret, P_hat, wealth, dist, last_a1 = carry
        TAU_B, ALPH_S, BETA_S, BETA_ACC, LAMN_S, TRUST_OFF = (
            calib[0], calib[1], calib[2], calib[3], calib[4], calib[5])
        kp, kc, ka1, kd, ke, kf, kg, kh, _ = random.split(tk, 9)

        k_idx = random.categorical(kp, LOGITS_PICK, axis=-1)
        ch = NB[AR, k_idx]; mut = (ch[ch] == AR)
        pid = jnp.minimum(AR, ch)
        ca = random.bernoulli(kc, 0.5, shape=(N,)).astype(jnp.float32)
        cp = ca[pid]; il = AR < ch
        pp = jnp.where(cp == 1.0, il, ~il)
        isP = pp & mut; isR = (~pp) & mut

        # --- stage 1 (engage/avoid) ---
        tau = TAU_B * (0.5 + O_p)
        w_eff = jnp.clip(w_base * (1.0 + CB) * (1.0 - dist * E.KAPPA_DECAY), 0.0, 1.0)
        # MB value from best over Q_offer's category 2 (fair = 5): use max over all 4 offers
        # as the "value of engaging" in the ultimatum sub-game; analogous to v5.
        best_prop = jnp.max(Q_off, axis=-1)     # (N, 2)
        best_send = jnp.max(Q_send, axis=-1)
        Q_MB = 0.5 * jnp.sum(P_hat * best_prop[:, None, :], axis=-1) \
             + 0.5 * jnp.sum(P_hat * best_send[:, None, :], axis=-1)
        Q_net = w_eff[:, None] * Q_MB + (1.0 - w_eff[:, None]) * Q_mf
        eng = E_bias - a_inhib; avd = theta_T + u_U
        per = phi * jnp.where(last_a1 == 0, 1.0, 0.0)
        L1 = tau[:, None] * jnp.stack(
            [Q_net[:, 0] + eng + per, Q_net[:, 1] + avd], axis=-1)
        L1 = jnp.clip(L1, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a1 = random.categorical(ka1, L1, axis=-1)
        cm = random.bernoulli(kh, E.P_COMMON, shape=(N,))
        s2 = jnp.where(cm, a1, 1 - a1)

        # --- role-conditioned heads ---
        # proposer offer
        db = SM_b + SN_b   # positive bias toward selfish (index 0)
        ab_off = jnp.stack([db, db*0.5, jnp.zeros_like(db), -db*0.25], axis=-1)
        qs_off = Q_off[AR, s2, :]
        L_off = jnp.clip(beta_s[:, None] * (qs_off + ab_off), -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a_off = random.categorical(kd, L_off, axis=-1)

        # sender amount
        ab_send = jnp.stack([db, db*0.3, jnp.zeros_like(db), -db*0.3], axis=-1)
        qs_send = Q_send[AR, s2, :]
        L_send = jnp.clip(beta_s[:, None] * (qs_send + ab_send), -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a_send = random.categorical(ke, L_send, axis=-1)

        # responder accept/reject (logits from Q_acc, one call per pair)
        qs_acc = Q_acc[AR, s2, :]
        L_acc = beta_s[:, None] * qs_acc
        a_acc = random.categorical(kf, L_acc, axis=-1)   # 0=accept, 1=reject

        # trustee return multiplier
        ab_ret = jnp.stack([-db, jnp.zeros_like(db), jnp.zeros_like(db), db*0.5], axis=-1)
        qs_ret = Q_ret[AR, s2, :]
        L_ret = jnp.clip(beta_s[:, None] * (qs_ret + ab_ret), -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a_ret = random.categorical(kg, L_ret, axis=-1)

        # --- role-specific values ---
        offer_vals = jnp.array([2.0, 4.0, 5.0, 7.0])
        send_vals  = jnp.array([0.0, 3.0, 6.0, 9.0])
        return_mult = jnp.array([0.0, 1.0, 2.0, 3.0])

        my_off  = offer_vals[a_off]
        my_send = send_vals[a_send]
        my_rm   = jnp.maximum(0.0, return_mult[a_ret] - TRUST_OFF)

        # partner's decisions
        rec_off  = my_off[ch]
        rec_send = my_send[ch]
        rec_rm   = my_rm[ch]
        partner_reject = a_acc[ch]  # 1 = reject
        my_reject = a_acc

        # --- Fehr-Schmidt utility (with scales) ---
        alpha_e = ALPH_S * lambda_e
        beta_g  = BETA_S  * A_e
        lam_n   = LAMN_S  * lambda_n

        # proposer's offer is accepted if partner does not reject AND partner is responder
        partner_accepts_me = (partner_reject == 0) & isP
        # my offer acceptance: responder decisions
        my_acceptance = ((my_reject == 0) & isR).astype(jnp.float32)

        # --- payoffs ---
        r_ultiP = jnp.where(partner_accepts_me, E.E_ENDOWMENT - my_off, 0.0)
        r_ultiR = jnp.where(my_acceptance >= 1.0, rec_off, 0.0)
        # trust: sender gets -send + TRUST_K * partner_return_mult * send
        r_ts = jnp.where(isR, -my_send + E.TRUST_K * rec_rm * my_send, 0.0)
        r_tt = jnp.where(isP, E.TRUST_K * rec_send * (1.0 - my_rm / E.TRUST_K), 0.0)

        raw = r_ultiP + r_ultiR + r_ts + r_tt
        rew = jnp.clip(raw * omega * (1.0 + sal + 0.05 * mood),
                       -E.REWARD_CLIP, E.REWARD_CLIP)
        inter = (a1 == 0) & (a1[ch] == 0) & mut
        rew = jnp.where(inter, rew, 0.0)
        um = inter.astype(jnp.float32)

        # --- Q updates on the role-appropriate head ---
        # Each agent updates the head corresponding to their role this tick.
        # Proposer: Q_off. Sender: Q_send. Responder: Q_acc. Trustee: Q_ret.
        # Same reward signal for all (reward is individual).
        q_off_v = Q_off[AR, s2, a_off]
        upd_off = jnp.where(isP, um, 0.0)
        Q_off_n = Q_off.at[AR, s2, a_off].set(jnp.clip(
            q_off_v + upd_off * lr * (rew - q_off_v), -E.GRAD_CLIP, E.GRAD_CLIP))

        q_send_v = Q_send[AR, s2, a_send]
        upd_send = jnp.where(isR, um, 0.0)
        Q_send_n = Q_send.at[AR, s2, a_send].set(jnp.clip(
            q_send_v + upd_send * lr * (rew - q_send_v), -E.GRAD_CLIP, E.GRAD_CLIP))

        q_acc_v = Q_acc[AR, s2, a_acc]
        upd_acc = jnp.where(isR, um, 0.0)
        Q_acc_n = Q_acc.at[AR, s2, a_acc].set(jnp.clip(
            q_acc_v + upd_acc * lr * (rew - q_acc_v), -E.GRAD_CLIP, E.GRAD_CLIP))

        q_ret_v = Q_ret[AR, s2, a_ret]
        upd_ret = jnp.where(isP, um, 0.0)
        Q_ret_n = Q_ret.at[AR, s2, a_ret].set(jnp.clip(
            q_ret_v + upd_ret * lr * (rew - q_ret_v), -E.GRAD_CLIP, E.GRAD_CLIP))

        # MF head (shared stage-1)
        qmv = Q_mf[AR, a1]; vn = jnp.max(Q_mf, axis=-1)
        Q_mfn = Q_mf.at[AR, a1].set(jnp.clip(
            qmv + um * lr * (rew + gamma * vn - qmv), -E.GRAD_CLIP, E.GRAD_CLIP))

        # P_hat
        oh = jnp.stack([1.0-s2.astype(jnp.float32), s2.astype(jnp.float32)], axis=-1)
        Pat = P_hat[AR, a1, :]
        Pu = Pat + um[:, None] * eta_vol[:, None] * (oh - Pat)
        Pu = jnp.clip(Pu, 1e-4, 1.0); Pu = Pu / jnp.sum(Pu, axis=-1, keepdims=True)
        P_hatn = P_hat.at[AR, a1, :].set(Pu)

        # distress
        rej = isR & inter & (my_reject == 1)
        dn = (E.DISTRESS_DECAY * dist + E.DISTRESS_PULL * baseline
              + (1.0 + N_sens * E.N_DISTRESS) * rej.astype(jnp.float32) * 0.05)
        wn = jnp.maximum(wealth + rew, 0.0)

        # capture
        captured = jnp.stack([
            jnp.where(inter & isP, my_off, 0.0),                     # 0
            jnp.where(inter & isR, (my_reject == 0).astype(jnp.float32), 0.0),  # 1
            jnp.where(inter & isR, 1.0, 0.0),                        # 2
            jnp.where(inter & isP, 1.0, 0.0),                        # 3
            jnp.where(inter & isR, my_send, 0.0),                    # 4
            jnp.where(inter & isR, 1.0, 0.0),                        # 5
            jnp.where(inter & isR, rec_rm * my_send, 0.0),           # 6
            jnp.where(inter & isP, rec_send, 0.0),                   # 7
            jnp.where(inter & isP, my_rm * rec_send, 0.0),           # 8
            jnp.where(inter & isP, 1.0, 0.0),                        # 9
        ], axis=-1)

        return (calib, Q_mfn, Q_off_n, Q_send_n, Q_acc_n, Q_ret_n,
                P_hatn, wn, dn, a1), captured

    @jax.jit
    def _scan(carry, ks):
        return lax.scan(step, carry, ks)

    @jax.jit
    def simulate(calib_vec):
        keys = random.split(random.PRNGKey(SEED), T)
        carry0 = (calib_vec, Q_MF0, Q_OFFER0, Q_SEND0, Q_ACC0, Q_RET0,
                  P_HAT0, W0, baseline, LAST0)
        final, cap = _scan(carry0, keys)
        wealth = final[7]

        c0 = cap[:, :, 0]; c1 = cap[:, :, 1]; c2 = cap[:, :, 2]
        c4 = cap[:, :, 4]; c5 = cap[:, :, 5]
        c7 = cap[:, :, 7]; c8 = cap[:, :, 8]; c9 = cap[:, :, 9]

        offers = c0.reshape(-1); offer_mask = offers > 0.0
        n_offers = jnp.maximum(jnp.sum(offer_mask), 1.0)
        ug_offer = jnp.where(jnp.sum(offer_mask) > 0,
                             jnp.sum(offers * offer_mask) / n_offers / E.E_ENDOWMENT,
                             0.0)

        resp_mask = c2.reshape(-1) > 0.0
        accepts = c1.reshape(-1)
        n_resp = jnp.maximum(jnp.sum(resp_mask), 1.0)
        ug_rej = jnp.where(jnp.sum(resp_mask) > 0,
                           1.0 - jnp.sum(accepts * resp_mask) / n_resp,
                           0.5)

        send_mask = c5.reshape(-1) > 0.0
        sends = c4.reshape(-1)
        n_send = jnp.maximum(jnp.sum(send_mask), 1.0)
        tg_send = jnp.where(jnp.sum(send_mask) > 0,
                            jnp.sum(sends * send_mask) / n_send / E.E_ENDOWMENT,
                            0.0)

        trustee_mask = c9.reshape(-1) > 0.0
        recv = c7.reshape(-1); ret = c8.reshape(-1)
        valid = trustee_mask & (recv > 0.0)
        n_valid = jnp.maximum(jnp.sum(valid), 1.0)
        ret_frac = jnp.where(jnp.sum(valid) > 0,
                             jnp.sum(ret * valid / jnp.where(recv > 0, recv, 1.0)) / n_valid / 3.0,
                             0.0)

        ws = jnp.sort(wealth); n = wealth.shape[0]
        cum = jnp.cumsum(ws); total = cum[-1]
        gini = jnp.where(total > 0, (n + 1 - 2.0 * jnp.sum(cum) / total) / n, 0.5)

        return jnp.stack([ug_offer, ug_rej, tg_send, ret_frac, gini])

    return simulate

print("Building 32k x 500 simulator ...", flush=True)
t0 = time.time()
SIM = build_simulator(32000, 500, SEED)
_ = SIM(jnp.array([2.0, 1.0, 1.0, 1.0, 1.0, 0.0], jnp.float32)).block_until_ready()
print(f"  compile wall = {time.time()-t0:.1f}s", flush=True)

def loss(calib_vec):
    plat = np.asarray(SIM(jnp.asarray(calib_vec, jnp.float32)))
    return float(np.sum(((plat - TARGETS) / SE) ** 2))

if __name__ == "__main__":
    def log(m): print(m, flush=True)
    log("=" * 72)
    log("PHASE 6c DECOUPLED — role-specific heads + joint calibration")
    log("=" * 72)
    log(f"Device: {jax.devices()[0]}")

    init = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    log(f"\nLoss at init = {loss(init):.3f}")

    log("\nDifferential evolution (32k x 500, popsize=12, maxiter=20) ...")
    t0 = time.time()
    gen = [0]
    def cb(xk, convergence):
        gen[0] += 1
        if gen[0] % 3 == 0:
            log(f"  gen {gen[0]:>3}  loss = {loss(xk):.4f}  wall = {time.time()-t0:.1f}s")
    result = differential_evolution(
        loss, BOUNDS, seed=SEED, maxiter=20, popsize=12,
        tol=1e-4, polish=False, workers=1, updating="deferred",
        init="sobol", callback=cb,
    )
    log(f"  DE wall = {time.time()-t0:.1f}s")
    log(f"  best loss = {result.fun:.4f}")
    log(f"  θ = {dict(zip(CALIB_NAMES, [float(v) for v in result.x]))}")

    plat = np.asarray(SIM(jnp.asarray(result.x, jnp.float32)))
    names = ["UG mean offer", "UG rejection", "TG mean send", "TG mean return", "Final Gini"]
    log("\n" + "=" * 72)
    log("VERIFICATION (32k agents × 500 ticks)")
    log("=" * 72)
    log(f"{'benchmark':22s} {'platform':>10} {'human':>10} {'SE':>8} {'|diff|':>8} {'verdict':>8}")
    log("-" * 72)
    n_pass = 0
    verif = {}
    for name, p, t, se in zip(names, plat, TARGETS, SE):
        d = abs(p - t); v = "PASS" if d < se else "FAIL"
        n_pass += int(v == "PASS")
        verif[name] = {"platform": float(p), "human": float(t),
                       "abs_diff": float(d), "se": float(se), "verdict": v}
        log(f"{name:22s} {p:>10.4f} {t:>10.4f} {se:>8.4f} {d:>8.4f} {v:>8s}")
    log("-" * 72)
    log(f"OVERALL: {n_pass}/5 PASS")

    with open(OUT / "p6c_results.json", "w") as f:
        json.dump({
            "calib": dict(zip(CALIB_NAMES, [float(v) for v in result.x])),
            "loss": float(result.fun),
            "verification": verif,
            "n_pass": n_pass,
        }, f, indent=2)
    log(f"Saved: {OUT / 'p6c_results.json'}")
