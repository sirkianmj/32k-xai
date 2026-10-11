# p6c_calibrate.py — Phase 6c: joint calibration against human benchmarks.
#
# Structural fixes over engine_v5:
#   1. dark_bias sign corrected (positive pushes toward selfish).
#   2. Fehr-Schmidt terms scaled: alpha_eff = ALPHA_SCALE * (D + lambda_ineq),
#      beta_guilt = BETA_SCALE * A.
#   3. Probabilistic acceptance: p_accept = sigmoid(BETA_ACCEPT * U_R).
#   4. Trustee return offset: return_mult = max(0, TRUST_RETURN_MULT[a2] - OFFSET).
#   5. TAU_BASE and LAM_NORM_SCALE exposed as calibration scalars.
#
# Calibration method: differential evolution over 6 scalars, loss = sum of
# ((platform - human)/SE)^2 over 5 benchmarks. Fast sim (5k x 200) during
# search, full sim (32k x 500) for final verification.
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
BOUNDS = [(0.5, 4.0), (0.2, 3.0), (0.2, 3.0),
          (0.1, 5.0), (0.0, 3.0), (-1.0, 2.0)]

TARGETS = np.array([0.40, 0.16, 0.502, 0.372, 0.40])   # 5 benchmarks
SE      = np.array([0.03, 0.03, 0.124, 0.114, 0.10])

_src = E.load_all(PARAMS)

def make_simulator(N, T):
    idx = np.random.default_rng(SEED).integers(0, len(_src["O"]), size=N)
    P = {k: jnp.asarray(v[idx]) for k, v in _src.items()}

    _CB       = P["C"] * E.C_BOOST
    _eta_vol  = P["eta_vol"]
    _E_bias   = P["E"] * E.E_ENGAGE
    _a_inhib  = P["a_inhib"] * E.A_INHIB
    _theta_T  = P["theta_T"] * E.T_THREAT
    _u_U      = P["u_U"] * E.U_UNCERT
    _SM_b     = P["S_M"] * E.SM_DEFECT
    _SP_b     = P["S_P"] * E.SP_PUNISH
    _SN_b     = P["S_N"] * E.SN_STATUS
    _omega    = P["omega_social"]
    _sal      = P["gamma_sal"] * E.SAL_GAIN
    _mood     = P["m_t"]
    _lambda_e = P["D"] + P["lambda_ineq"]
    _A_e      = P["A"]
    _lambda_n = P["lambda_norm"] * E.LAMBDA_NORM_A
    _N_sens   = P["N"]
    _lr       = P["alpha_R"]
    _beta_s   = P["beta_choice"]
    _phi      = P["phi_per"]
    _w_base   = P["w_MB"]
    _gamma    = 1.0/(1.0 + jnp.exp(jnp.clip(P["k_DD"], -10.0, 5.0)))
    _baseline = (P["general_distress"] + 0.3*P["DASS_Dep"]
                 + 0.3*P["DASS_Anx"] + 0.3*P["DASS_Str"]) / 1.9

    nb, _ = E.build_ws(n=N, k=6, p=0.1, seed=0)
    NB = jnp.asarray(nb, jnp.int32)
    AR = jnp.arange(N)
    LOGITS_PICK = jnp.where(NB >= 0, 0.0, -1e30)

    Q_MF0  = jnp.zeros((N, 2), jnp.float32)
    Q_S2_0 = jnp.zeros((N, 2, E.N_ACTIONS), jnp.float32)
    P_HAT0 = jnp.zeros((N, 2, 2), jnp.float32)
    P_HAT0 = P_HAT0.at[:, 0, 0].set(E.P_COMMON).at[:, 0, 1].set(1.0 - E.P_COMMON)
    P_HAT0 = P_HAT0.at[:, 1, 0].set(1.0 - E.P_COMMON).at[:, 1, 1].set(E.P_COMMON)
    W0     = jnp.full((N,), E.WEALTH_INIT, jnp.float32)
    LAST0  = jnp.full((N,), -1, jnp.int32)

    def step(carry, tk):
        calib, Q_mf, Q_s2, P_hat, wealth, dist, last_a1 = carry
        TAU_B, ALPH_S, BETA_S, BETA_ACC, LAMN_S, TRUST_OFF = (
            calib[0], calib[1], calib[2], calib[3], calib[4], calib[5])

        kp, kc, ka1, ka2, kr, ka_acc, _ = random.split(tk, 7)

        k_idx = random.categorical(kp, LOGITS_PICK, axis=-1)
        ch = NB[AR, k_idx]; mut = (ch[ch] == AR)
        pid = jnp.minimum(AR, ch)
        ca = random.bernoulli(kc, 0.5, shape=(N,)).astype(jnp.float32)
        cp = ca[pid]; il = AR < ch
        pp = jnp.where(cp == 1.0, il, ~il)
        isP = pp & mut; isR = (~pp) & mut

        # stage-1 with TAU_BASE from calib
        tau = TAU_B * (0.5 + P["O"])
        w_eff = jnp.clip(_w_base * (1.0 + _CB) * (1.0 - dist * E.KAPPA_DECAY),
                         0.0, 1.0)
        best = jnp.max(Q_s2, axis=-1)
        Q_MB = jnp.sum(P_hat * best[:, None, :], axis=-1)
        Q_net = w_eff[:, None] * Q_MB + (1.0 - w_eff[:, None]) * Q_mf
        eng = _E_bias - _a_inhib; avd = _theta_T + _u_U
        per = _phi * jnp.where(last_a1 == 0, 1.0, 0.0)
        L1 = tau[:, None] * jnp.stack(
            [Q_net[:, 0] + eng + per, Q_net[:, 1] + avd], axis=-1)
        L1 = jnp.clip(L1, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a1 = random.categorical(ka1, L1, axis=-1)

        cm = random.bernoulli(kr, E.P_COMMON, shape=(N,))
        s2 = jnp.where(cm, a1, 1 - a1)

        # stage-2 with corrected dark_bias sign (POSITIVE)
        qs = Q_s2[AR, s2, :]
        qs = jnp.where(qs < 0.0, qs * (1.0 - _SP_b[:, None] * 0.5), qs)
        db = _SM_b + _SN_b   # sign corrected
        ab = jnp.stack([db, db * 0.5, jnp.zeros_like(db), jnp.zeros_like(db)], axis=-1)
        L2 = _beta_s[:, None] * (qs + ab)
        L2 = jnp.clip(L2, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a2 = random.categorical(ka2, L2, axis=-1)

        my_off = E.ULTI_OFFERS[a2]
        rec_off = my_off[ch]

        # Fehr-Schmidt with scaling
        alpha_e = ALPH_S * _lambda_e
        beta_g  = BETA_S * _A_e
        lam_n   = LAMN_S * _lambda_n

        d_me = 2.0 * rec_off - E.E_ENDOWMENT
        rp_me = jnp.maximum(d_me, 0.0); rn_me = jnp.maximum(-d_me, 0.0)
        my_UR = rec_off - alpha_e * rn_me - beta_g * rp_me + lam_n

        # Probabilistic acceptance (sigmoid on U_R)
        p_accept = jax.nn.sigmoid(BETA_ACC * my_UR)
        acc_draw = random.bernoulli(ka_acc, p_accept).astype(jnp.float32)
        my_acc = acc_draw * isR.astype(jnp.float32)

        d_p = 2.0 * my_off - E.E_ENDOWMENT
        rp_p = jnp.maximum(d_p, 0.0); rn_p = jnp.maximum(-d_p, 0.0)
        p_UR = my_off - alpha_e[ch] * rn_p - beta_g[ch] * rp_p
        p_accept_prob = jax.nn.sigmoid(BETA_ACC * p_UR)
        partner_accept = (p_accept_prob * isP).astype(jnp.float32)

        # trust with trustee offset
        my_send = E.TRUST_SENDS[a2]
        rec_send = my_send[ch]
        my_rm = jnp.maximum(0.0, E.TRUST_RETURN_MULT[a2] - TRUST_OFF)
        rec_rm = jnp.maximum(0.0, E.TRUST_RETURN_MULT[a2][ch] - TRUST_OFF)

        r_ultiP = jnp.where(partner_accept >= 1.0, E.E_ENDOWMENT - my_off, 0.0)
        r_ultiR = jnp.where(my_acc >= 1.0, rec_off, 0.0)
        r_ts = jnp.where(isR, -my_send + E.TRUST_K * rec_rm * my_send, 0.0)
        r_tt = jnp.where(isP, E.TRUST_K * rec_send * (1.0 - my_rm / E.TRUST_K), 0.0)

        raw = r_ultiP + r_ultiR + r_ts + r_tt
        rew = jnp.clip(raw * _omega * (1.0 + _sal + 0.05 * _mood),
                       -E.REWARD_CLIP, E.REWARD_CLIP)
        inter = (a1 == 0) & (a1[ch] == 0) & mut
        rew = jnp.where(inter, rew, 0.0)
        um = inter.astype(jnp.float32)

        qsv = Q_s2[AR, s2, a2]
        Q_s2n = Q_s2.at[AR, s2, a2].set(jnp.clip(
            qsv + um * _lr * (rew - qsv), -E.GRAD_CLIP, E.GRAD_CLIP))
        qmv = Q_mf[AR, a1]; vn = jnp.max(Q_mf, axis=-1)
        Q_mfn = Q_mf.at[AR, a1].set(jnp.clip(
            qmv + um * _lr * (rew + _gamma * vn - qmv), -E.GRAD_CLIP, E.GRAD_CLIP))
        oh = jnp.stack([1.0-s2.astype(jnp.float32), s2.astype(jnp.float32)], axis=-1)
        Pat = P_hat[AR, a1, :]
        Pu = Pat + um[:, None] * _eta_vol[:, None] * (oh - Pat)
        Pu = jnp.clip(Pu, 1e-4, 1.0); Pu = Pu / jnp.sum(Pu, axis=-1, keepdims=True)
        P_hatn = P_hat.at[AR, a1, :].set(Pu)

        rej = isR & inter & (my_UR < 0.0)
        dn = (E.DISTRESS_DECAY * dist + E.DISTRESS_PULL * _baseline
              + (1.0 + _N_sens * E.N_DISTRESS) * rej.astype(jnp.float32) * 0.05)
        wn = jnp.maximum(wealth + rew, 0.0)

        captured = jnp.stack([
            jnp.where(inter & isP, my_off, 0.0),
            jnp.where(inter & isR, my_acc, 0.0),
            jnp.where(inter & isR, 1.0, 0.0),
            jnp.where(inter & isP, 1.0, 0.0),
            jnp.where(inter & is_trust_sender if False else (inter & isR), my_send, 0.0),
            jnp.where(inter & isR, 1.0, 0.0),
            jnp.where(inter & isR, rec_rm * my_send, 0.0),
            jnp.where(inter & isP, rec_send, 0.0),
            jnp.where(inter & isP, my_rm * rec_send, 0.0),
            jnp.where(inter & isP, 1.0, 0.0),
        ], axis=-1)
        return (calib, Q_mfn, Q_s2n, P_hatn, wn, dn, a1), captured

    @jax.jit
    def run(carry, ks):
        return lax.scan(step, carry, ks)

    def simulate(calib_vec):
        keys = random.split(random.PRNGKey(SEED), T)
        carry0 = (jnp.asarray(calib_vec, jnp.float32),
                  Q_MF0, Q_S2_0, P_HAT0, W0, _baseline, LAST0)
        final, cap = run(carry0, keys)
        jax.block_until_ready(cap)
        return np.asarray(cap), np.asarray(final[4])

    return simulate

def extract_benchmarks(cap):
    # cap shape (T, N, 10)
    offers = cap[:, :, 0].reshape(-1); offers = offers[offers > 0]
    n_off = max(len(offers), 1)

    resp_mask = cap[:, :, 2].reshape(-1) > 0
    accepts   = cap[:, :, 1].reshape(-1)[resp_mask]
    n_resp    = max(len(accepts), 1)
    rej_rate  = 1.0 - accepts.mean() if len(accepts) > 0 else 0.5

    sender_mask = cap[:, :, 5].reshape(-1) > 0
    sends       = cap[:, :, 4].reshape(-1)[sender_mask]
    n_send      = max(len(sends), 1)
    send_frac   = sends.mean() / E.E_ENDOWMENT if len(sends) > 0 else 0.0

    trustee_mask = cap[:, :, 9].reshape(-1) > 0
    recv = cap[:, :, 7].reshape(-1)[trustee_mask]
    ret  = cap[:, :, 8].reshape(-1)[trustee_mask]
    valid = recv > 0
    ret_frac = (ret[valid] / recv[valid]).mean() / 3.0 if valid.any() else 0.0

    return (offers.mean() / E.E_ENDOWMENT if len(offers) > 0 else 0.0,
            rej_rate, send_frac, ret_frac, 0.0)  # Gini filled in by caller

def loss(calib_vec):
    sim = make_simulator(5000, 200)
    cap, wealth = sim(calib_vec)
    ug_off, ug_rej, tg_send, tg_ret, _ = extract_benchmarks(cap)
    # Gini from final wealth
    ws = np.sort(wealth); n = len(ws); cum = np.cumsum(ws); total = cum[-1]
    gini = (n + 1 - 2 * np.sum(cum) / total) / n if total > 0 else 0.5

    plat = np.array([ug_off, ug_rej, tg_send, tg_ret, gini])
    return float(np.sum(((plat - TARGETS) / SE) ** 2))

if __name__ == "__main__":
    def log(m): print(m, flush=True)
    log("=" * 72)
    log("PHASE 6c — Joint calibration vs. human benchmarks")
    log("=" * 72)
    log(f"Device: {jax.devices()[0]}")

    # Quick eval at init point
    init = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    log(f"\nLoss at init {init.tolist()} = {loss(init):.3f}")

    # Calibrate
    log("\nRunning differential evolution ...")
    t0 = time.time()
    result = differential_evolution(
        loss, BOUNDS, seed=SEED, maxiter=40, popsize=12,
        tol=1e-4, polish=True, workers=1, updating="deferred",
        init="sobol",
    )
    log(f"  wall = {time.time()-t0:.1f}s")
    log(f"  best loss = {result.fun:.4f}")
    log(f"  best θ    = {dict(zip(CALIB_NAMES, result.x.tolist()))}")

    # Full verification at 32k x 500
    log("\nVerifying at 32k x 500 ...")
    sim_full = make_simulator(32000, 500)
    t0 = time.time()
    cap_full, wealth_full = sim_full(result.x)
    log(f"  wall = {time.time()-t0:.2f}s")

    ug_off, ug_rej, tg_send, tg_ret, _ = extract_benchmarks(cap_full)
    ws = np.sort(wealth_full); n = len(ws); cum = np.cumsum(ws); total = cum[-1]
    gini = (n + 1 - 2*np.sum(cum)/total) / n if total > 0 else 0.5

    log("\n" + "=" * 72)
    log("VERIFICATION (32k agents × 500 ticks, calibrated θ)")
    log("=" * 72)
    log(f"{'benchmark':22s} {'platform':>10} {'human':>10} {'SE':>8} {'|diff|':>8} {'verdict':>8}")
    log("-" * 72)
    names = ["UG mean offer", "UG rejection", "TG mean send", "TG mean return", "Final Gini"]
    plat  = [ug_off, ug_rej, tg_send, tg_ret, gini]
    n_pass = 0
    for name, p, t, se in zip(names, plat, TARGETS, SE):
        d = abs(p - t); v = "PASS" if d < se else "FAIL"
        n_pass += int(v == "PASS")
        log(f"{name:22s} {p:>10.4f} {t:>10.4f} {se:>8.4f} {d:>8.4f} {v:>8s}")
    log("-" * 72)
    log(f"OVERALL: {n_pass}/5 PASS")

    with open(OUT / "p6c_results.json", "w") as f:
        json.dump({
            "calib": dict(zip(CALIB_NAMES, result.x.tolist())),
            "loss": result.fun,
            "verification": {n: {"platform": float(p), "human": float(t),
                                 "se": float(s), "verdict": "PASS" if abs(p-t)<s else "FAIL"}
                             for n, p, t, s in zip(names, plat, TARGETS, SE)},
            "n_pass": n_pass,
        }, f, indent=2)
    log(f"Saved: {OUT / 'p6c_results.json'}")
