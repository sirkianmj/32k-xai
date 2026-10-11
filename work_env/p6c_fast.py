# p6c_fast.py — Phase 6c: fast joint calibration.
#
# Structure:
#   - All static setup (copula slice, WS graph, initial tensors) done ONCE
#     at module load — never inside the loss function.
#   - simulate() is JIT'd; the whole compute graph (agents + step + aggregation)
#     is compiled once and reused for every calibration vector.
#   - Two simulators: small (3000x100) for search, full (32k x 500) for verify.
#
# Calibration: differential evolution over 6 scalars.
# Target: < 3 min total wall on GTX 1660 Ti.
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
BOUNDS = [(0.5, 3.0), (0.2, 3.0), (0.2, 3.0),
          (0.1, 5.0), (0.0, 3.0), (-0.5, 1.5)]

TARGETS = np.array([0.40, 0.16, 0.502, 0.372, 0.40])
SE      = np.array([0.05, 0.05, 0.130, 0.120, 0.12])

# ---- Load copula ONCE ----
_SRC = E.load_all(PARAMS)
_N_ALL = len(_SRC["O"])

# =============================================================================
# Build a simulator for a given N and T. Called ONCE per (N, T) pair at
# module import time. Never inside a loss or calibration loop.
# =============================================================================
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

        tau = TAU_B * (0.5 + O_p)
        w_eff = jnp.clip(w_base * (1.0 + CB) * (1.0 - dist * E.KAPPA_DECAY), 0.0, 1.0)
        best = jnp.max(Q_s2, axis=-1)
        Q_MB = jnp.sum(P_hat * best[:, None, :], axis=-1)
        Q_net = w_eff[:, None] * Q_MB + (1.0 - w_eff[:, None]) * Q_mf
        eng = E_bias - a_inhib; avd = theta_T + u_U
        per = phi * jnp.where(last_a1 == 0, 1.0, 0.0)
        L1 = tau[:, None] * jnp.stack(
            [Q_net[:, 0] + eng + per, Q_net[:, 1] + avd], axis=-1)
        L1 = jnp.clip(L1, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a1 = random.categorical(ka1, L1, axis=-1)

        cm = random.bernoulli(kr, E.P_COMMON, shape=(N,))
        s2 = jnp.where(cm, a1, 1 - a1)

        qs = Q_s2[AR, s2, :]
        qs = jnp.where(qs < 0.0, qs * (1.0 - SP_b[:, None] * 0.5), qs)
        db = SM_b + SN_b
        ab = jnp.stack([db, db*0.5, jnp.zeros_like(db), jnp.zeros_like(db)], axis=-1)
        L2 = beta_s[:, None] * (qs + ab)
        L2 = jnp.clip(L2, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a2 = random.categorical(ka2, L2, axis=-1)

        my_off = E.ULTI_OFFERS[a2]
        rec_off = my_off[ch]
        alpha_e = ALPH_S * lambda_e
        beta_g  = BETA_S  * A_e
        lam_n   = LAMN_S  * lambda_n

        d_me = 2.0 * rec_off - E.E_ENDOWMENT
        rp_me = jnp.maximum(d_me, 0.0); rn_me = jnp.maximum(-d_me, 0.0)
        my_UR = rec_off - alpha_e * rn_me - beta_g * rp_me + lam_n
        p_accept = jax.nn.sigmoid(BETA_ACC * my_UR)
        acc_draw = random.bernoulli(ka_acc, p_accept).astype(jnp.float32)
        my_acc = acc_draw * isR.astype(jnp.float32)

        d_p = 2.0 * my_off - E.E_ENDOWMENT
        rp_p = jnp.maximum(d_p, 0.0); rn_p = jnp.maximum(-d_p, 0.0)
        p_UR = my_off - alpha_e[ch] * rn_p - beta_g[ch] * rp_p
        p_accept_prob = jax.nn.sigmoid(BETA_ACC * p_UR)
        partner_accept = (p_accept_prob * isP).astype(jnp.float32)

        my_send = E.TRUST_SENDS[a2]
        rec_send = my_send[ch]
        my_rm = jnp.maximum(0.0, E.TRUST_RETURN_MULT[a2] - TRUST_OFF)
        rec_rm = jnp.maximum(0.0, E.TRUST_RETURN_MULT[a2][ch] - TRUST_OFF)

        r_ultiP = jnp.where(partner_accept >= 1.0, E.E_ENDOWMENT - my_off, 0.0)
        r_ultiR = jnp.where(my_acc >= 1.0, rec_off, 0.0)
        r_ts = jnp.where(isR, -my_send + E.TRUST_K * rec_rm * my_send, 0.0)
        r_tt = jnp.where(isP, E.TRUST_K * rec_send * (1.0 - my_rm / E.TRUST_K), 0.0)

        raw = r_ultiP + r_ultiR + r_ts + r_tt
        rew = jnp.clip(raw * omega * (1.0 + sal + 0.05 * mood),
                       -E.REWARD_CLIP, E.REWARD_CLIP)
        inter = (a1 == 0) & (a1[ch] == 0) & mut
        rew = jnp.where(inter, rew, 0.0)
        um = inter.astype(jnp.float32)

        qsv = Q_s2[AR, s2, a2]
        Q_s2n = Q_s2.at[AR, s2, a2].set(jnp.clip(
            qsv + um * lr * (rew - qsv), -E.GRAD_CLIP, E.GRAD_CLIP))
        qmv = Q_mf[AR, a1]; vn = jnp.max(Q_mf, axis=-1)
        Q_mfn = Q_mf.at[AR, a1].set(jnp.clip(
            qmv + um * lr * (rew + gamma * vn - qmv), -E.GRAD_CLIP, E.GRAD_CLIP))
        oh = jnp.stack([1.0-s2.astype(jnp.float32), s2.astype(jnp.float32)], axis=-1)
        Pat = P_hat[AR, a1, :]
        Pu = Pat + um[:, None] * eta_vol[:, None] * (oh - Pat)
        Pu = jnp.clip(Pu, 1e-4, 1.0); Pu = Pu / jnp.sum(Pu, axis=-1, keepdims=True)
        P_hatn = P_hat.at[AR, a1, :].set(Pu)

        rej = isR & inter & (my_UR < 0.0)
        dn = (E.DISTRESS_DECAY * dist + E.DISTRESS_PULL * baseline
              + (1.0 + N_sens * E.N_DISTRESS) * rej.astype(jnp.float32) * 0.05)
        wn = jnp.maximum(wealth + rew, 0.0)

        # 10-channel capture
        captured = jnp.stack([
            jnp.where(inter & isP, my_off, 0.0),                    # 0 offer
            jnp.where(inter & isR, my_acc, 0.0),                    # 1 accept
            jnp.where(inter & isR, 1.0, 0.0),                       # 2 responder tick
            jnp.where(inter & isP, 1.0, 0.0),                       # 3 proposer tick
            jnp.where(inter & isR, my_send, 0.0),                   # 4 send
            jnp.where(inter & isR, 1.0, 0.0),                       # 5 sender tick
            jnp.where(inter & isR, rec_rm * my_send, 0.0),          # 6 received back
            jnp.where(inter & isP, rec_send, 0.0),                  # 7 received as trustee
            jnp.where(inter & isP, my_rm * rec_send, 0.0),          # 8 returned by me
            jnp.where(inter & isP, 1.0, 0.0),                       # 9 trustee tick
        ], axis=-1)
        return (calib, Q_mfn, Q_s2n, P_hatn, wn, dn, a1), captured

    @jax.jit
    def _scan(carry, ks):
        return lax.scan(step, carry, ks)

    @jax.jit
    def simulate(calib_vec):
        keys = random.split(random.PRNGKey(SEED), T)
        carry0 = (calib_vec, Q_MF0, Q_S2_0, P_HAT0, W0, baseline, LAST0)
        final, cap = _scan(carry0, keys)
        # final[4] is wealth
        wealth = final[4]

        # Extract benchmarks in JIT
        # cap: (T, N, 10)
        c0 = cap[:, :, 0]; c1 = cap[:, :, 1]; c2 = cap[:, :, 2]
        c4 = cap[:, :, 4]; c5 = cap[:, :, 5]
        c7 = cap[:, :, 7]; c8 = cap[:, :, 8]; c9 = cap[:, :, 9]

        # UG mean offer
        offers = c0.reshape(-1); offer_mask = offers > 0.0
        n_offers = jnp.maximum(jnp.sum(offer_mask), 1.0)
        ug_offer = jnp.where(jnp.sum(offer_mask) > 0,
                             jnp.sum(offers * offer_mask) / n_offers / E.E_ENDOWMENT,
                             0.0)

        # UG rejection
        resp_mask = c2.reshape(-1) > 0.0
        accepts = c1.reshape(-1)
        n_resp = jnp.maximum(jnp.sum(resp_mask), 1.0)
        n_accept = jnp.sum(accepts * resp_mask)
        ug_rej = jnp.where(jnp.sum(resp_mask) > 0,
                           1.0 - n_accept / n_resp,
                           0.5)

        # TG send fraction
        send_mask = c5.reshape(-1) > 0.0
        sends = c4.reshape(-1)
        n_send = jnp.maximum(jnp.sum(send_mask), 1.0)
        tg_send = jnp.where(jnp.sum(send_mask) > 0,
                            jnp.sum(sends * send_mask) / n_send / E.E_ENDOWMENT,
                            0.0)

        # TG return fraction (per-event returned/received, then mean; /3)
        trustee_mask = c9.reshape(-1) > 0.0
        recv = c7.reshape(-1); ret = c8.reshape(-1)
        valid = trustee_mask & (recv > 0.0)
        n_valid = jnp.maximum(jnp.sum(valid), 1.0)
        ret_frac = jnp.where(jnp.sum(valid) > 0,
                             jnp.sum(ret * valid / jnp.where(recv > 0, recv, 1.0)) / n_valid / 3.0,
                             0.0)

        # Gini
        ws = jnp.sort(wealth); n = wealth.shape[0]
        cum = jnp.cumsum(ws); total = cum[-1]
        gini = jnp.where(total > 0,
                         (n + 1 - 2.0 * jnp.sum(cum) / total) / n,
                         0.5)

        return jnp.stack([ug_offer, ug_rej, tg_send, ret_frac, gini])

    return simulate

# ---- Build simulators ONCE ----
print("Building small simulator (3000 x 100) ...", flush=True)
t0 = time.time()
SIM_SMALL = build_simulator(3000, 100, SEED)
_ = SIM_SMALL(jnp.array([2.0, 1.0, 1.0, 1.0, 1.0, 0.0], jnp.float32)).block_until_ready()
print(f"  compile wall = {time.time()-t0:.1f}s", flush=True)

def loss(calib_vec):
    plat = np.asarray(SIM_SMALL(jnp.asarray(calib_vec, jnp.float32)))
    return float(np.sum(((plat - TARGETS) / SE) ** 2))

if __name__ == "__main__":
    def log(m): print(m, flush=True)

    log("=" * 72)
    log("PHASE 6c FAST — Joint calibration vs. human benchmarks")
    log("=" * 72)
    log(f"Device: {jax.devices()[0]}")
    log(f"Targets: {dict(zip(['UG offer', 'UG rej', 'TG send', 'TG ret', 'Gini'], TARGETS.tolist()))}")

    init = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    log(f"\nLoss at init = {loss(init):.3f}")

    log("\nRunning differential evolution (popsize=8, maxiter=15, no polish) ...")
    t0 = time.time()
    call_count = [0]
    def cb(xk, convergence):
        call_count[0] += 1
        if call_count[0] % 5 == 0:
            log(f"  gen {call_count[0]:>3}  best loss = {loss(xk):.4f}  "
                f"wall = {time.time()-t0:.1f}s")
    result = differential_evolution(
        loss, BOUNDS, seed=SEED, maxiter=15, popsize=8,
        tol=1e-3, polish=False, workers=1, updating="deferred",
        init="sobol", callback=cb,
    )
    log(f"  DE wall = {time.time()-t0:.1f}s")
    log(f"  best loss = {result.fun:.4f}")
    log(f"  θ = {dict(zip(CALIB_NAMES, result.x.tolist()))}")

    # Verify at 32k x 500
    log("\nBuilding full simulator (32k x 500) ...")
    t0 = time.time()
    SIM_FULL = build_simulator(32000, 500, SEED)
    _ = SIM_FULL(jnp.array(result.x, jnp.float32)).block_until_ready()
    log(f"  compile wall = {time.time()-t0:.1f}s")

    t0 = time.time()
    plat = np.asarray(SIM_FULL(jnp.asarray(result.x, jnp.float32)))
    log(f"  run wall = {time.time()-t0:.2f}s")

    names = ["UG mean offer", "UG rejection", "TG mean send", "TG mean return", "Final Gini"]
    log("\n" + "=" * 72)
    log("VERIFICATION (32k agents × 500 ticks, calibrated θ)")
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
            "calib": dict(zip(CALIB_NAMES, result.x.tolist())),
            "loss": float(result.fun),
            "verification": verif,
            "n_pass": n_pass,
        }, f, indent=2)
    log(f"Saved: {OUT / 'p6c_results.json'}")
