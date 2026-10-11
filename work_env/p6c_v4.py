# p6c_v4.py — Phase 6c v4: per-head rewards + FS accept + 9 calibration scalars.
#
# Changes from p6c_v3:
#   - Accept decision now includes the Fehr-Schmidt utility term from
#     Master Spec Part V Module 1. This restores spec compliance and
#     enables rejections of low offers (which pure RL cannot produce
#     because rejecting yields 0 reward so Q_acc[reject] never learns).
#   - 9th calibration scalar LAMBDA_FS controls the weight of the FS term.
#     LAMBDA_FS = 0 recovers pure RL; larger values restore normative
#     inequity aversion.
#
# References:
#   Fehr & Schmidt (1999), QJE — FS utility
#   Sanfey et al. (2003), Science — insula activity predicts rejection
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

CALIB_NAMES = ["TAU_BASE", "BETA_OFF", "BETA_SEND", "BETA_ACC",
               "BETA_RET", "BIAS_OFF", "BIAS_SEND", "BIAS_RET", "LAMBDA_FS"]
BOUNDS = [(0.3, 4.0), (0.3, 6.0), (0.3, 6.0), (0.3, 8.0),
          (0.3, 6.0), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0), (0.0, 8.0)]

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

    nb, _ = E.build_ws(n=N, k=6, p=0.1, seed=0)
    NB = jnp.asarray(nb, jnp.int32)
    AR = jnp.arange(N)
    LOGITS_PICK = jnp.where(NB >= 0, 0.0, -1e30)

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

    def step(carry, tk):
        calib, Q_mf, Q_off, Q_send, Q_acc, Q_ret, P_hat, wealth, dist, last_a1 = carry
        TAU_B, B_OFF, B_SEND, B_ACC, B_RET, BI_OFF, BI_SEND, BI_RET, LAMB_FS = (
            calib[0], calib[1], calib[2], calib[3],
            calib[4], calib[5], calib[6], calib[7], calib[8])
        kp, kc, ka1, kd, ke, kf, kg, kh, _ = random.split(tk, 9)

        k_idx = random.categorical(kp, LOGITS_PICK, axis=-1)
        ch = NB[AR, k_idx]; mut = (ch[ch] == AR)
        ca = random.bernoulli(kc, 0.5, shape=(N,)).astype(jnp.float32)
        cp = ca[jnp.minimum(AR, ch)]
        il = AR < ch
        pp = jnp.where(cp == 1.0, il, ~il)
        isP = pp & mut; isR = (~pp) & mut

        # ---- stage 1: engage/avoid ----
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

        # ---- Fehr-Schmidt params (used in accept AND offer) ----
        alpha_e = P["ALPHA_SCALE_PLACEHOLDER"] if False else P["D"] + P["lambda_ineq"]
        beta_g  = P["A"]
        # For head-specific softmax, use calibrated scalars on the utilities.

        # ---- 4 role heads ----
        bias_off = jnp.stack([BI_OFF, -BI_OFF/2, jnp.zeros_like(BI_OFF), -BI_OFF/2], axis=-1)
        L_off = jnp.clip(B_OFF * (Q_off[AR, s2, :] + bias_off), -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a_off = random.categorical(kd, L_off, axis=-1)

        bias_send = jnp.stack([-BI_SEND, jnp.zeros_like(BI_SEND), jnp.zeros_like(BI_SEND), BI_SEND], axis=-1)
        L_send = jnp.clip(B_SEND * (Q_send[AR, s2, :] + bias_send), -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a_send = random.categorical(ke, L_send, axis=-1)

        # ---- accept decision: RL + Fehr-Schmidt ----
        # rec_off is the offer I receive as responder (from my partner).
        # But rec_off is defined below after my_off. We compute the offer head first.
        # The order in v3 was: offer head, then send head, then accept.
        # We now interleave: after my_off is chosen, we can compute partner's offer
        # which we would receive. But that's partner's choice, not ours.
        # Simplification: compute U_accept from the offer that this agent RECEIVES.
        # The received offer is my_off[ch] because each agent's offer goes to their partner.
        # We compute this below, after my_off is chosen.

        # Placeholder: we will compute a_acc AFTER we know my_off and rec_off.
        # Save the logits structure but defer the categorical draw.

        # ---- offer/send values (needed before accept) ----
        my_off = OFFER_VALS[a_off]
        my_send = SEND_VALS[a_send]
        rec_off = my_off[ch]   # the offer I receive from my partner

        # Fehr-Schmidt utility of accepting rec_off (fair ref = 5):
        fair_ref = E.E_ENDOWMENT * 0.5
        d_me = rec_off - fair_ref
        inequity_neg = jnp.maximum(-d_me, 0.0)   # offer < fair (disadvantageous)
        inequity_pos = jnp.maximum( d_me, 0.0)   # offer > fair (advantageous)
        U_accept = rec_off - alpha_e * inequity_neg - beta_g * inequity_pos

        L_acc_accept = B_ACC * Q_acc[AR, s2, 0] + LAMB_FS * U_accept
        L_acc_reject = B_ACC * Q_acc[AR, s2, 1]
        L_acc = jnp.clip(jnp.stack([L_acc_accept, L_acc_reject], axis=-1),
                         -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a_acc = random.categorical(kf, L_acc, axis=-1)  # 0=accept, 1=reject

        # ---- return head ----
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

        # ---- per-head rewards ----
        r_off = jnp.where(partner_accepts_me, E.E_ENDOWMENT - my_off, 0.0)
        r_acc = jnp.where(my_acceptance >= 1.0, rec_off, 0.0)
        r_send = jnp.where(isR, -my_send + E.TRUST_K * rec_rm * my_send, 0.0)
        r_ret = jnp.where(isP, E.TRUST_K * rec_send * (1.0 - my_rm / E.TRUST_K), 0.0)

        mult = omega * (1.0 + sal + 0.05 * mood)
        r_off_m = jnp.clip(r_off * mult, -E.REWARD_CLIP, E.REWARD_CLIP)
        r_acc_m = jnp.clip(r_acc * mult, -E.REWARD_CLIP, E.REWARD_CLIP)
        r_send_m = jnp.clip(r_send * mult, -E.REWARD_CLIP, E.REWARD_CLIP)
        r_ret_m = jnp.clip(r_ret * mult, -E.REWARD_CLIP, E.REWARD_CLIP)

        inter = (a1 == 0) & (a1[ch] == 0) & mut
        um = inter.astype(jnp.float32)

        # ---- per-head Q updates ----
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

        # MF from total reward
        total_r = jnp.clip((r_off + r_acc + r_send + r_ret) * mult,
                           -E.REWARD_CLIP, E.REWARD_CLIP)
        total_r = jnp.where(inter, total_r, 0.0)
        qmv = Q_mf[AR, a1]; vn = jnp.max(Q_mf, axis=-1)
        Q_mfn = Q_mf.at[AR, a1].set(jnp.clip(
            qmv + um * lr * (total_r + gamma * vn - qmv), -E.GRAD_CLIP, E.GRAD_CLIP))

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
        wn = jnp.maximum(wealth + total_r, 0.0)

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
        ], axis=-1)

        return (calib, Q_mfn, Q_off_n, Q_send_n, Q_acc_n, Q_ret_n,
                P_hatn, wn, dn, a1), cap

    @jax.jit
    def _scan(carry, ks):
        return lax.scan(step, carry, ks)

    @jax.jit
    def simulate(calib_vec):
        keys = random.split(random.PRNGKey(SEED), T)
        carry0 = (calib_vec, Q_MF0, Q_OFF0, Q_SEND0, Q_ACC0, Q_RET0,
                  P_HAT0, W0, baseline, LAST0)
        final, cap = _scan(carry0, keys)
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

        return jnp.stack([ug_offer, ug_rej, tg_send, ret_frac, gini])

    return simulate

print("Building small simulator (8000 x 250) ...", flush=True)
t0 = time.time()
SIM_SMALL = build_simulator(8000, 250, SEED)
_ = SIM_SMALL(jnp.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0], jnp.float32)).block_until_ready()
print(f"  compile wall = {time.time()-t0:.1f}s", flush=True)

print("Building full simulator (32k x 500) ...", flush=True)
t0 = time.time()
SIM_FULL = build_simulator(32000, 500, SEED)
_ = SIM_FULL(jnp.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0], jnp.float32)).block_until_ready()
print(f"  compile wall = {time.time()-t0:.1f}s", flush=True)

def loss(calib_vec):
    plat = np.asarray(SIM_SMALL(jnp.asarray(calib_vec, jnp.float32)))
    return float(np.sum(((plat - TARGETS) / SE) ** 2))

if __name__ == "__main__":
    def log(m): print(m, flush=True)
    log("=" * 72)
    log("PHASE 6c v4 — FS accept + 9 calibration scalars")
    log("=" * 72)
    log(f"Device: {jax.devices()[0]}")

    init = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0], dtype=np.float32)
    log(f"\nLoss at init = {loss(init):.3f}")

    log("\nDifferential evolution (popsize=12, maxiter=20, no polish) ...")
    t0 = time.time()
    gen = [0]
    def cb(xk, convergence):
        gen[0] += 1
        if gen[0] % 3 == 0:
            log(f"  gen {gen[0]:>3}  loss = {float(convergence):.4f}  wall = {time.time()-t0:.1f}s")
    result = differential_evolution(
        loss, BOUNDS, seed=SEED, maxiter=20, popsize=12,
        tol=1e-5, polish=False, workers=1, updating="deferred",
        init="sobol", callback=cb,
    )
    log(f"  DE wall = {time.time()-t0:.1f}s")
    log(f"  best loss = {result.fun:.4f}")
    log(f"  θ = {dict(zip(CALIB_NAMES, [round(float(v),4) for v in result.x]))}")

    plat = np.asarray(SIM_FULL(jnp.asarray(result.x, jnp.float32)))
    names = ["UG mean offer", "UG rejection", "TG mean send", "TG mean return", "Final Gini"]
    log("\n" + "=" * 72)
    log("VERIFICATION (32k × 500, calibrated θ)")
    log("=" * 72)
    log(f"{'benchmark':22s} {'platform':>10} {'human':>10} {'SE':>8} {'|diff|':>8} {'verdict':>8}")
    log("-" * 72)
    n_pass = 0; verif = {}
    for name, p, t, se in zip(names, plat, TARGETS, SE):
        d = abs(p - t); v = "PASS" if d < se else "FAIL"
        n_pass += int(v == "PASS")
        verif[name] = {"platform": float(p), "human": float(t),
                       "abs_diff": float(d), "se": float(se), "verdict": v}
        log(f"{name:22s} {p:>10.4f} {t:>10.4f} {se:>8.4f} {d:>8.4f} {v:>8s}")
    log("-" * 72)
    log(f"OVERALL: {n_pass}/5 PASS")

    with open(OUT / "p6c_results.json", "w") as f:
        json.dump({"calib": dict(zip(CALIB_NAMES, [float(v) for v in result.x])),
                   "loss": float(result.fun), "verification": verif,
                   "n_pass": n_pass}, f, indent=2)
    log(f"Saved: {OUT / 'p6c_results.json'}")
