# p6b_human_benchmark.py — Phase 6b: Human benchmark validation.
#
# Runs engine_v5 at 32k agents, extracts the behavioural statistics that
# correspond to published human-game benchmarks, and compares.
#
# Benchmarks (from meta-analyses):
#
#   Ultimatum Game:
#     - mean offer fraction     : 0.40  (Oosterbeek et al. 2004, N=37 papers, 75 results)
#     - rejection rate          : 0.16  (Oosterbeek et al. 2004)
#     - cross-cultural range    : 0.26–0.58  (Henrich et al. 2001, 15 societies)
#
#   Trust Game:
#     - mean fraction sent      : 0.502 (SE=0.124)  (Johnson & Mislin 2011, N=162)
#     - mean fraction returned  : 0.372 (SE=0.114)  (Johnson & Mislin 2011)
#
#   Public Goods Game:
#     - mean contribution       : ~0.40 of endowment (Bucciarelli et al. 2025)
#
#   Wealth inequality (from UG payoffs):
#     - baseline Gini           : 0.127 rising to 0.175  (Bucciarelli et al. 2025)
#     - spatial-UG Gini band    : 0.4–0.6  (Alves & Monteiro 2019, Latin America / Africa)
#
# PASS criterion per benchmark: platform value within ±1 combined SE of the
# empirical value, where combined SE = sqrt(SE_emp^2 + SE_sim^2) and
# SE_sim ~ 1/sqrt(N_effective).
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import time, json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax
import engine_v5 as E

OUT = Path(__file__).parent
PARAMS = Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"
N = 32000
T = 500
SEED = 42

# -----------------------------------------------------------------------------
# Load copula
# -----------------------------------------------------------------------------
_src = E.load_all(PARAMS)
_idx = np.arange(N)
_P = {k: jnp.asarray(v[_idx]) for k, v in _src.items()}

_CB       = _P["C"]
_eta_vol  = _P["eta_vol"]
_E_bias   = _P["E"] * E.E_ENGAGE
_a_inhib  = _P["a_inhib"] * E.A_INHIB
_theta_T  = _P["theta_T"] * E.T_THREAT
_u_U      = _P["u_U"] * E.U_UNCERT
_SM_b     = _P["S_M"] * E.SM_DEFECT
_SP_b     = _P["S_P"] * E.SP_PUNISH
_SN_b     = _P["S_N"] * E.SN_STATUS
_omega    = _P["omega_social"]
_lam_norm = _P["lambda_norm"] * E.LAMBDA_NORM_A
_sal      = _P["gamma_sal"] * E.SAL_GAIN
_mood     = _P["m_t"]
_alpha_e  = _P["D"] + _P["lambda_ineq"]
_beta_e   = _P["A"]
_gamma    = 1.0/(1.0 + jnp.exp(jnp.clip(_P["k_DD"], -10.0, 5.0)))
_baseline = (_P["general_distress"] + 0.3*_P["DASS_Dep"]
             + 0.3*_P["DASS_Anx"] + 0.3*_P["DASS_Str"]) / 1.9
_N_sens   = _P["N"]
_lr       = _P["alpha_R"]
_beta_s   = _P["beta_choice"]
_phi      = _P["phi_per"]
_w_base   = _P["w_MB"]

_nb, _ = E.build_ws(n=N, k=6, p=0.1, seed=0)
_NB = jnp.asarray(_nb, jnp.int32)
_AR = jnp.arange(N)
_LOGITS_PICK = jnp.where(_NB >= 0, 0.0, -1e30)

Q_MF0  = jnp.zeros((N, 2), jnp.float32)
Q_S2_0 = jnp.zeros((N, 2, E.N_ACTIONS), jnp.float32)
P_HAT0 = jnp.zeros((N, 2, 2), jnp.float32)
P_HAT0 = P_HAT0.at[:, 0, 0].set(E.P_COMMON).at[:, 0, 1].set(1.0 - E.P_COMMON)
P_HAT0 = P_HAT0.at[:, 1, 0].set(1.0 - E.P_COMMON).at[:, 1, 1].set(E.P_COMMON)
W0     = jnp.full((N,), E.WEALTH_INIT, jnp.float32)
LAST0  = jnp.full((N,), -1, jnp.int32)

# -----------------------------------------------------------------------------
# Step with full behavioural capture
# -----------------------------------------------------------------------------
def step(carry, tk):
    Q_mf, Q_s2, P_hat, wealth, dist, last_a1 = carry
    kp, kc, ka1, ka2, kr, _ = random.split(tk, 6)

    k_idx = random.categorical(kp, _LOGITS_PICK, axis=-1)
    ch = _NB[_AR, k_idx]
    mut = (ch[ch] == _AR)
    pid = jnp.minimum(_AR, ch)
    ca = random.bernoulli(kc, 0.5, shape=(N,)).astype(jnp.float32)
    cp = ca[pid]; il = _AR < ch
    pp = jnp.where(cp == 1.0, il, ~il)
    isP = pp & mut; isR = (~pp) & mut

    w_eff = jnp.clip(_w_base * (1.0 + _CB * E.C_BOOST) *
                     (1.0 - dist * E.KAPPA_DECAY), 0.0, 1.0)
    best = jnp.max(Q_s2, axis=-1)
    Q_MB = jnp.sum(P_hat * best[:, None, :], axis=-1)
    Q_net = w_eff[:, None] * Q_MB + (1.0 - w_eff[:, None]) * Q_mf

    eng = _E_bias - _a_inhib
    avd = _theta_T + _u_U
    per = _phi * jnp.where(last_a1 == 0, 1.0, 0.0)
    L1 = _beta_s[:, None] * jnp.stack(
        [Q_net[:, 0] + eng + per, Q_net[:, 1] + avd], axis=-1)
    L1 = jnp.clip(L1, -E.LOGIT_CLIP, E.LOGIT_CLIP)
    a1 = random.categorical(ka1, L1, axis=-1)

    cm = random.bernoulli(kr, E.P_COMMON, shape=(N,))
    s2 = jnp.where(cm, a1, 1 - a1)

    qs = Q_s2[_AR, s2, :]
    qs = jnp.where(qs < 0.0, qs * (1.0 - _SP_b[:, None] * 0.5), qs)
    db = -_SM_b - _SN_b
    ab = jnp.stack([db, db * 0.5, jnp.zeros_like(db), jnp.zeros_like(db)], axis=-1)
    L2 = _beta_s[:, None] * (qs + ab)
    L2 = jnp.clip(L2, -E.LOGIT_CLIP, E.LOGIT_CLIP)
    a2 = random.categorical(ka2, L2, axis=-1)

    # ultimatum
    my_off = E.ULTI_OFFERS[a2]
    rec_off = my_off[ch]
    d_me = 2.0 * rec_off - E.E_ENDOWMENT
    rp_me = jnp.maximum(d_me, 0.0); rn_me = jnp.maximum(-d_me, 0.0)
    my_UR = rec_off - _alpha_e * rn_me - _beta_e * rp_me + _lam_norm
    my_acc = ((my_UR >= 0.0) & isR).astype(jnp.float32)

    d_p = 2.0 * my_off - E.E_ENDOWMENT
    rp_p = jnp.maximum(d_p, 0.0); rn_p = jnp.maximum(-d_p, 0.0)
    p_UR = my_off - _alpha_e[ch] * rn_p - _beta_e[ch] * rp_p
    p_acc = ((p_UR >= 0.0) & isP).astype(jnp.float32)

    # trust
    is_trust_sender = isR
    is_trust_trustee = isP
    my_send = E.TRUST_SENDS[a2]
    rec_send = my_send[ch]
    my_rm = E.TRUST_RETURN_MULT[a2]
    rec_rm = my_rm[ch]
    returned_by_partner = rec_rm * my_send          # amount partner returns to me
    received_back_by_me = returned_by_partner       # alias
    returned_by_me = my_rm * rec_send               # amount I return to partner

    r_ultiP = jnp.where(p_acc >= 1.0, E.E_ENDOWMENT - my_off, 0.0)
    r_ultiR = jnp.where(my_acc >= 1.0, rec_off, 0.0)
    r_ts = jnp.where(is_trust_sender, -my_send + E.TRUST_K * rec_rm * my_send, 0.0)
    r_tt = jnp.where(is_trust_trustee, E.TRUST_K * rec_send * (1.0 - my_rm / E.TRUST_K), 0.0)

    raw = r_ultiP + r_ultiR + r_ts + r_tt
    rew = jnp.clip(raw * _omega * (1.0 + _sal + 0.05 * _mood),
                   -E.REWARD_CLIP, E.REWARD_CLIP)
    inter = (a1 == 0) & (a1[ch] == 0) & mut
    rew = jnp.where(inter, rew, 0.0)
    um = inter.astype(jnp.float32)

    # Q updates
    qsv = Q_s2[_AR, s2, a2]
    Q_s2n = Q_s2.at[_AR, s2, a2].set(jnp.clip(
        qsv + um * _lr * (rew - qsv), -E.GRAD_CLIP, E.GRAD_CLIP))
    qmv = Q_mf[_AR, a1]; vn = jnp.max(Q_mf, axis=-1)
    Q_mfn = Q_mf.at[_AR, a1].set(jnp.clip(
        qmv + um * _lr * (rew + _gamma * vn - qmv), -E.GRAD_CLIP, E.GRAD_CLIP))
    oh = jnp.stack([1.0 - s2.astype(jnp.float32), s2.astype(jnp.float32)], axis=-1)
    Pat = P_hat[_AR, a1, :]
    Pu = Pat + um[:, None] * _eta_vol[:, None] * (oh - Pat)
    Pu = jnp.clip(Pu, 1e-4, 1.0); Pu = Pu / jnp.sum(Pu, axis=-1, keepdims=True)
    P_hatn = P_hat.at[_AR, a1, :].set(Pu)

    rej = isR & inter & (my_UR < 0.0)
    dn = (E.DISTRESS_DECAY * dist + E.DISTRESS_PULL * _baseline
          + (1.0 + _N_sens * E.N_DISTRESS) * rej.astype(jnp.float32) * 0.05)
    wn = jnp.maximum(wealth + rew, 0.0)

    # per-tick capture: only count ticks where interaction happened
    captured = jnp.stack([
        jnp.where(inter, my_off, 0.0),               # 0 ultimatum offer
        jnp.where(inter & isR, my_acc, 0.0),          # 1 accepted? (only responders)
        jnp.where(inter & isR, 1.0, 0.0),             # 2 responder tick?
        jnp.where(inter & isP, my_off, 0.0),          # 3 proposer offer (for offer distribution)
        jnp.where(inter & isP, p_acc, 0.0),           # 4 partner accepted my offer?
        jnp.where(inter & is_trust_sender, my_send, 0.0),       # 5 trust send
        jnp.where(inter & is_trust_sender, 1.0, 0.0),           # 6 sender tick?
        jnp.where(inter & is_trust_sender, rec_rm * my_send, 0.0),  # 7 received back
        jnp.where(inter & is_trust_trustee, rec_send, 0.0),     # 8 amount received as trustee
        jnp.where(inter & is_trust_trustee, my_rm * rec_send, 0.0),  # 9 amount I returned
        jnp.where(inter & is_trust_trustee, 1.0, 0.0),          # 10 trustee tick?
    ], axis=-1)  # (N, 11)

    return (Q_mfn, Q_s2n, P_hatn, wn, dn, a1), captured

@jax.jit
def run(carry, ks):
    return lax.scan(step, carry, ks)

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    def log(m): print(m, flush=True)
    log("=" * 72)
    log("PHASE 6b — HUMAN BENCHMARK VALIDATION")
    log("=" * 72)
    log(f"Device: {jax.devices()[0]}  N={N}  T={T}")

    log("\nRunning 32k agents x 500 ticks ...")
    t0 = time.time()
    keys = random.split(random.PRNGKey(SEED), T)
    carry0 = (Q_MF0, Q_S2_0, P_HAT0, W0, _baseline, LAST0)
    final, cap = run(carry0, keys)
    jax.block_until_ready(cap)
    wall = time.time() - t0
    log(f"  wall = {wall:.2f}s")

    cap = np.asarray(cap)              # (T, N, 11)
    log(f"  captured shape = {cap.shape}")

    # Final wealth
    wealth = np.asarray(final[3])

    # -------------------------------------------------------------------------
    # Aggregate the captured statistics
    # -------------------------------------------------------------------------
    # Ultimatum proposer offers (only engaged ticks with isP)
    offers_all = cap[:, :, 3].reshape(-1)               # proposer offers
    offers_all = offers_all[offers_all > 0]              # only non-zero (engaged)
    n_offers = len(offers_all)

    # Ultimatum responder decisions
    resp_mask = cap[:, :, 2].reshape(-1) > 0
    accept_vals = cap[:, :, 1].reshape(-1)[resp_mask]
    n_resp = len(accept_vals)
    rejection_rate = 1.0 - accept_vals.mean() if n_resp > 0 else float("nan")

    # Trust sender sends
    sender_mask = cap[:, :, 6].reshape(-1) > 0
    sends = cap[:, :, 5].reshape(-1)[sender_mask]
    n_sends = len(sends)
    mean_send_frac = sends.mean() / E.E_ENDOWMENT if n_sends > 0 else float("nan")

    # Trust returns
    trustee_mask = cap[:, :, 10].reshape(-1) > 0
    amt_recv = cap[:, :, 8].reshape(-1)[trustee_mask]
    amt_ret  = cap[:, :, 9].reshape(-1)[trustee_mask]
    n_trust = len(amt_recv)
    # Return fraction = returned / received (skip when received == 0)
    valid = amt_recv > 0
    mean_return_frac = (amt_ret[valid] / amt_recv[valid]).mean() if valid.any() else float("nan")

    # Gini of final wealth
    ws = np.sort(wealth)
    n = len(ws); cum = np.cumsum(ws); total = cum[-1]
    gini = (n + 1 - 2 * np.sum(cum) / total) / n

    log("\n" + "=" * 72)
    log("PLATFORM OUTPUTS")
    log("=" * 72)
    log(f"  Ultimatum proposer offers   n={n_offers:>8d}  mean offer = {offers_all.mean():.3f} of E")
    log(f"  Ultimatum responder ticks   n={n_resp:>8d}  rejection rate = {rejection_rate:.4f}")
    log(f"  Trust send events           n={n_sends:>8d}  mean send = {mean_send_frac:.4f} of E")
    log(f"  Trust return events         n={n_trust:>8d}  mean return frac = {mean_return_frac:.4f}")
    log(f"  Final wealth                Gini = {gini:.4f}")

    # -------------------------------------------------------------------------
    # Benchmarks
    # -------------------------------------------------------------------------
    ug_offer_frac = offers_all.mean() / E.E_ENDOWMENT
    tg_return_frac = mean_return_frac / 3.0   # mult in {0,1,2,3} → fraction returned

    log(f"\n  UG offer as fraction: {ug_offer_frac:.4f}")
    log(f"  TG return as fraction: {tg_return_frac:.4f}")

    BENCHMARKS = [
        ("UG mean offer (frac)", 0.40,  0.03, ug_offer_frac),
        ("UG rejection rate",    0.16,  0.03, rejection_rate),
        ("TG mean send (frac)",  0.502, 0.124, mean_send_frac),
        ("TG mean return (frac)",0.372, 0.114, tg_return_frac),
        ("Final wealth Gini",    0.40,  0.10, gini),
    ]

    log("\n" + "=" * 72)
    log("COMPARISON vs. PUBLISHED HUMAN BENCHMARKS")
    log("=" * 72)
    log(f"{'benchmark':22s} {'platform':>10} {'human':>10} {'emp_SE':>8} "
        f"{'combined_SE':>12} {'|diff|':>8} {'verdict':>10}")
    log("-" * 72)

    SE_sim = 1.0 / np.sqrt(max(n_resp, 1))
    results = {}
    for name, emp, emp_se, platform in BENCHMARKS:
        diff = abs(platform - emp)
        combined_se = np.sqrt(emp_se**2 + SE_sim**2)
        passed = diff < 1.0 * combined_se
        verdict = "PASS" if passed else "FAIL"
        results[name] = {"platform": float(platform), "human": emp,
                         "abs_diff": float(diff), "combined_se": float(combined_se),
                         "verdict": verdict}
        log(f"{name:22s} {platform:>10.4f} {emp:>10.4f} {emp_se:>8.4f} "
            f"{combined_se:>12.4f} {diff:>8.4f} {verdict:>10s}")

    n_pass = sum(1 for r in results.values() if r["verdict"] == "PASS")
    log("-" * 72)
    log(f"OVERALL: {n_pass}/{len(BENCHMARKS)} benchmarks PASS "
        f"(|platform - human| < 1 combined SE)")

    with open(OUT / "p6b_results.json", "w") as f:
        json.dump({"platform": {
            "ug_mean_offer": float(offers_all.mean()),
            "ug_rejection_rate": float(rejection_rate),
            "tg_mean_send": float(mean_send_frac),
            "tg_mean_return": float(mean_return_frac),
            "final_gini": float(gini),
            "n_offers": int(n_offers), "n_responses": int(n_resp),
            "n_sends": int(n_sends), "n_trustees": int(n_trust),
        }, "benchmarks": results, "wall_s": wall,
           "config": {"N": N, "T": T, "SEED": SEED}}, f, indent=2)
    log(f"\nSaved: {OUT / 'p6b_results.json'}")
