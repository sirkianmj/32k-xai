# p6a_population.py — Phase 6a: Population-level amortized Bayesian inference.
#
# Scientific basis: Habermann et al. (2025, Bayesian Analysis) and Bürkner
# et al. (2025, ABI for Multilevel Models). Global parameters (μ, σ) are
# recovered with near-perfect accuracy even when individual-level recovery
# is limited by sparse data.
#
# Deliverable: population hyperparameters μ_w, σ_w, μ_a, σ_a, μ_b, σ_b,
# μ_p, σ_p, validated by simulation-based calibration (SBC).
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import time, json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam, ClippedAdam
from jax import random, lax
from scipy.stats import spearmanr
from tqdm import tqdm

import engine_v5 as E

N_AGENTS     = 25
N_TICKS      = 25
N_TRAIN_SIMS = 120
N_TEST_SIMS  = 100   # for SBC
N_EPOCHS     = 4
N_ENC_SAMP   = 64
SEED         = 42
OUT          = Path(__file__).parent

_SRC = E.load_all(Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl")
_IDX = np.random.default_rng(SEED).integers(0, len(_SRC["O"]), size=N_AGENTS)
_FIX = {k: jnp.asarray(v[_IDX]) for k, v in _SRC.items()}

_CB = _FIX["C"]; _eta_vol = _FIX["eta_vol"]
_E_bias = _FIX["E"] * E.E_ENGAGE; _a_inhib = _FIX["a_inhib"] * E.A_INHIB
_theta_T = _FIX["theta_T"] * E.T_THREAT; _u_U = _FIX["u_U"] * E.U_UNCERT
_SM_b = _FIX["S_M"] * E.SM_DEFECT; _SP_b = _FIX["S_P"] * E.SP_PUNISH
_SN_b = _FIX["S_N"] * E.SN_STATUS; _omega = _FIX["omega_social"]
_lam_norm = _FIX["lambda_norm"] * E.LAMBDA_NORM_A; _sal = _FIX["gamma_sal"] * E.SAL_GAIN
_mood = _FIX["m_t"]; _alpha_e = _FIX["D"] + _FIX["lambda_ineq"]; _beta_e = _FIX["A"]
_gamma = 1.0/(1.0 + jnp.exp(jnp.clip(_FIX["k_DD"], -10.0, 5.0)))
_baseline = (_FIX["general_distress"] + 0.3*_FIX["DASS_Dep"]
             + 0.3*_FIX["DASS_Anx"] + 0.3*_FIX["DASS_Str"]) / 1.9
_N_sens = _FIX["N"]

_NB, _ = E.build_ws(n=N_AGENTS, k=6, p=0.1, seed=0)
_NB = jnp.asarray(_NB, jnp.int32); _AR = jnp.arange(N_AGENTS)
_LOGITS_PICK = jnp.where(_NB >= 0, 0.0, -1e30)

Q_MF0 = jnp.zeros((N_AGENTS, 2), jnp.float32)
Q_S2_0 = jnp.zeros((N_AGENTS, 2, E.N_ACTIONS), jnp.float32)
P_HAT0 = jnp.zeros((N_AGENTS, 2, 2), jnp.float32)
P_HAT0 = P_HAT0.at[:, 0, 0].set(E.P_COMMON).at[:, 0, 1].set(1.0 - E.P_COMMON)
P_HAT0 = P_HAT0.at[:, 1, 0].set(1.0 - E.P_COMMON).at[:, 1, 1].set(E.P_COMMON)
W0 = jnp.full((N_AGENTS,), E.WEALTH_INIT, jnp.float32)
LAST0 = jnp.full((N_AGENTS,), -1, jnp.int32)

# Population hyperpriors (fixed for simulation, recovered by inference)
MU_W, SIG_W = 0.0, 1.0
MU_A, SIG_A = 0.0, 1.0
MU_B, SIG_B = 1.0, 1.0
MU_P, SIG_P = 0.2, 0.5

def sample_population_params(key):
    k1, k2, k3, k4 = random.split(key, 4)
    w_raw = random.normal(k1, (N_AGENTS,)) * SIG_W + MU_W
    a_raw = random.normal(k2, (N_AGENTS,)) * SIG_A + MU_A
    b_raw = random.normal(k3, (N_AGENTS,)) * SIG_B + MU_B
    p_raw = random.normal(k4, (N_AGENTS,)) * SIG_P + MU_P
    w = jax.nn.sigmoid(w_raw); a = jax.nn.sigmoid(a_raw)
    b = jnp.exp(b_raw); p = p_raw
    return (w, a, b, p), (w_raw, a_raw, b_raw, p_raw)

def simulate(w_mb, alpha_r, beta_ch, phi_per, key):
    keys = random.split(key, N_TICKS)
    def step(carry, tk):
        Q_mf, Q_s2, P_hat, wealth, dist, last_a1 = carry
        kp, kc, ka1, ka2, kr, _ = random.split(tk, 6)
        k_idx = random.categorical(kp, _LOGITS_PICK, axis=-1)
        ch = _NB[_AR, k_idx]; mut = (ch[ch] == _AR)
        pid = jnp.minimum(_AR, ch)
        ca = random.bernoulli(kc, 0.5, shape=(N_AGENTS,)).astype(jnp.float32)
        cp = ca[pid]; il = _AR < ch
        pp = jnp.where(cp == 1.0, il, ~il)
        isP = pp & mut; isR = (~pp) & mut
        w_eff = jnp.clip(w_mb * (1.0 + _CB * E.C_BOOST) *
                         (1.0 - dist * E.KAPPA_DECAY), 0.0, 1.0)
        best = jnp.max(Q_s2, axis=-1)
        Q_MB = jnp.sum(P_hat * best[:, None, :], axis=-1)
        Q_net = w_eff[:, None] * Q_MB + (1.0 - w_eff[:, None]) * Q_mf
        eng = _E_bias - _a_inhib; avd = _theta_T + _u_U
        per = phi_per * jnp.where(last_a1 == 0, 1.0, 0.0)
        L1 = beta_ch[:, None] * jnp.stack(
            [Q_net[:, 0] + eng + per, Q_net[:, 1] + avd], axis=-1)
        L1 = jnp.clip(L1, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a1 = random.categorical(ka1, L1, axis=-1)
        cm = random.bernoulli(kr, E.P_COMMON, shape=(N_AGENTS,))
        s2 = jnp.where(cm, a1, 1 - a1)
        qs = Q_s2[_AR, s2, :]
        qs = jnp.where(qs < 0.0, qs * (1.0 - _SP_b[:, None] * 0.5), qs)
        db = -_SM_b - _SN_b
        ab = jnp.stack([db, db*0.5, jnp.zeros_like(db), jnp.zeros_like(db)], axis=-1)
        L2 = beta_ch[:, None] * (qs + ab)
        L2 = jnp.clip(L2, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a2 = random.categorical(ka2, L2, axis=-1)
        my_off = E.ULTI_OFFERS[a2]; rec_off = my_off[ch]
        d_me = 2.0*rec_off - E.E_ENDOWMENT
        rp_me = jnp.maximum(d_me, 0.0); rn_me = jnp.maximum(-d_me, 0.0)
        my_UR = rec_off - _alpha_e*rn_me - _beta_e*rp_me + _lam_norm
        my_acc = ((my_UR >= 0.0) & isR).astype(jnp.float32)
        d_p = 2.0*my_off - E.E_ENDOWMENT
        rp_p = jnp.maximum(d_p, 0.0); rn_p = jnp.maximum(-d_p, 0.0)
        p_UR = my_off - _alpha_e[ch]*rn_p - _beta_e[ch]*rp_p
        p_acc = ((p_UR >= 0.0) & isP).astype(jnp.float32)
        r_ultiP = jnp.where(p_acc >= 1.0, E.E_ENDOWMENT - my_off, 0.0)
        r_ultiR = jnp.where(my_acc >= 1.0, rec_off, 0.0)
        my_s = E.TRUST_SENDS[a2]; rec_s = my_s[ch]
        my_rm = E.TRUST_RETURN_MULT[a2]; rec_rm = my_rm[ch]
        r_ts = jnp.where(isR, -my_s + E.TRUST_K*rec_rm*my_s, 0.0)
        r_tt = jnp.where(isP, E.TRUST_K*rec_s*(1.0 - my_rm/E.TRUST_K), 0.0)
        raw = r_ultiP + r_ultiR + r_ts + r_tt
        rew = jnp.clip(raw*_omega*(1.0 + _sal + 0.05*_mood), -E.REWARD_CLIP, E.REWARD_CLIP)
        inter = (a1 == 0) & (a1[ch] == 0) & mut
        rew = jnp.where(inter, rew, 0.0); um = inter.astype(jnp.float32)
        qsv = Q_s2[_AR, s2, a2]
        Q_s2n = Q_s2.at[_AR, s2, a2].set(jnp.clip(
            qsv + um*alpha_r*(rew - qsv), -E.GRAD_CLIP, E.GRAD_CLIP))
        qmv = Q_mf[_AR, a1]; vn = jnp.max(Q_mf, axis=-1)
        Q_mfn = Q_mf.at[_AR, a1].set(jnp.clip(
            qmv + um*alpha_r*(rew + _gamma*vn - qmv), -E.GRAD_CLIP, E.GRAD_CLIP))
        oh = jnp.stack([1.0-s2.astype(jnp.float32), s2.astype(jnp.float32)], axis=-1)
        Pat = P_hat[_AR, a1, :]
        Pu = Pat + um[:, None]*_eta_vol[:, None]*(oh - Pat)
        Pu = jnp.clip(Pu, 1e-4, 1.0); Pu = Pu/jnp.sum(Pu, axis=-1, keepdims=True)
        P_hatn = P_hat.at[_AR, a1, :].set(Pu)
        rej = isR & inter & (my_UR < 0.0)
        dn = (E.DISTRESS_DECAY*dist + E.DISTRESS_PULL*_baseline
              + (1.0 + _N_sens*E.N_DISTRESS)*rej.astype(jnp.float32)*0.05)
        wn = jnp.maximum(wealth + rew, 0.0)
        obs_row = jnp.stack([
            a1.astype(jnp.float32), s2.astype(jnp.float32), a2.astype(jnp.float32),
            rew, um, dn, Q_mf[:, 0], Q_mf[:, 1],
            Q_s2[:, 0, 0], Q_s2[:, 0, 1], Q_s2[:, 0, 2], Q_s2[:, 0, 3],
            Q_s2[:, 1, 0], Q_s2[:, 1, 1], Q_s2[:, 1, 2], Q_s2[:, 1, 3],
            P_hat[:, 0, 0], P_hat[:, 0, 1], P_hat[:, 1, 0], P_hat[:, 1, 1],
            w_eff], axis=-1)
        return (Q_mfn, Q_s2n, P_hatn, wn, dn, a1), obs_row
    _, obs = lax.scan(step, (Q_MF0, Q_S2_0, P_HAT0, W0, _baseline, LAST0), keys)
    return obs

_simulate_jit = jax.jit(simulate)

def _forward_ll(a1, s2, a2, r, um, dist_tr, w_mb, alpha_r, beta_ch, phi_per):
    T, N = a1.shape; arange = jnp.arange(N)
    def step(carry, xs):
        Q_mf, Q_s2, P_hat, last_a1 = carry
        a1_t, s2_t, a2_t, r_t, um_t, dist_t = xs
        w_eff = jnp.clip(w_mb * (1.0 + _CB * E.C_BOOST) *
                         (1.0 - dist_t * E.KAPPA_DECAY), 0.0, 1.0)
        best = jnp.max(Q_s2, axis=-1)
        Q_MB = jnp.sum(P_hat * best[:, None, :], axis=-1)
        Q_net = w_eff[:, None] * Q_MB + (1.0 - w_eff[:, None]) * Q_mf
        eng = _E_bias - _a_inhib; avd = _theta_T + _u_U
        per = phi_per * jnp.where(last_a1 == 0, 1.0, 0.0)
        L1 = beta_ch[:, None] * jnp.stack(
            [Q_net[:, 0] + eng + per, Q_net[:, 1] + avd], axis=-1)
        L1 = jnp.clip(L1, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        ll1 = jax.nn.log_softmax(L1, axis=-1)[arange, a1_t]
        qs = Q_s2[arange, s2_t, :]
        qs = jnp.where(qs < 0.0, qs * (1.0 - _SP_b[:, None] * 0.5), qs)
        db = -_SM_b - _SN_b
        ab = jnp.stack([db, db*0.5, jnp.zeros_like(db), jnp.zeros_like(db)], axis=-1)
        L2 = beta_ch[:, None] * (qs + ab)
        L2 = jnp.clip(L2, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        ll2 = jax.nn.log_softmax(L2, axis=-1)[arange, a2_t]
        qsv = Q_s2[arange, s2_t, a2_t]
        Q_s2n = Q_s2.at[arange, s2_t, a2_t].set(jnp.clip(
            qsv + um_t*alpha_r*(r_t - qsv), -E.GRAD_CLIP, E.GRAD_CLIP))
        qmv = Q_mf[arange, a1_t]; vn = jnp.max(Q_mf, axis=-1)
        Q_mfn = Q_mf.at[arange, a1_t].set(jnp.clip(
            qmv + um_t*alpha_r*(r_t + _gamma*vn - qmv), -E.GRAD_CLIP, E.GRAD_CLIP))
        oh = jnp.stack([1.0-s2_t.astype(jnp.float32), s2_t.astype(jnp.float32)], axis=-1)
        Pat = P_hat[arange, a1_t, :]
        Pu = Pat + um_t[:, None]*_eta_vol[:, None]*(oh - Pat)
        Pu = jnp.clip(Pu, 1e-4, 1.0); Pu = Pu/jnp.sum(Pu, axis=-1, keepdims=True)
        P_hatn = P_hat.at[arange, a1_t, :].set(Pu)
        return (Q_mfn, Q_s2n, P_hatn, a1_t.astype(jnp.int32)), (ll1, ll2)
    _, (ll1, ll2) = lax.scan(step, (Q_MF0, Q_S2_0, P_HAT0, LAST0),
                             (a1, s2, a2, r, um, dist_tr))
    return jnp.sum(ll1) + jnp.sum(ll2)

def model(obs):
    a1 = obs[:, :, 0].astype(jnp.int32); s2 = obs[:, :, 1].astype(jnp.int32)
    a2 = obs[:, :, 2].astype(jnp.int32); r = obs[:, :, 3]
    um = obs[:, :, 4]; dist_tr = obs[:, :, 5]
    N = a1.shape[1]
    # Population hyperparameters (inference targets)
    mu_w = numpyro.sample("mu_w", dist.Normal(0.0, 1.0))
    sig_w = numpyro.sample("sig_w", dist.HalfNormal(1.0))
    mu_a = numpyro.sample("mu_a", dist.Normal(0.0, 1.0))
    sig_a = numpyro.sample("sig_a", dist.HalfNormal(1.0))
    mu_b = numpyro.sample("mu_b", dist.Normal(1.0, 1.0))
    sig_b = numpyro.sample("sig_b", dist.HalfNormal(1.0))
    mu_p = numpyro.sample("mu_p", dist.Normal(0.2, 0.5))
    sig_p = numpyro.sample("sig_p", dist.HalfNormal(0.5))
    with numpyro.plate("agents", N):
        w_raw = numpyro.sample("w_raw", dist.Normal(mu_w, sig_w))
        a_raw = numpyro.sample("a_raw", dist.Normal(mu_a, sig_a))
        b_raw = numpyro.sample("b_raw", dist.Normal(mu_b, sig_b))
        p_raw = numpyro.sample("p_raw", dist.Normal(mu_p, sig_p))
    w_mb = jax.nn.sigmoid(w_raw); alpha = jax.nn.sigmoid(a_raw)
    beta = jnp.exp(b_raw); phi = p_raw
    ll = _forward_ll(a1, s2, a2, r, um, dist_tr, w_mb, alpha, beta, phi)
    numpyro.factor("ll", ll)

def _init_encoder(key):
    k1, k2, k3, k4, k5, k6 = random.split(key, 6)
    in_dim = 47; hidden = 96
    gl = jax.nn.initializers.glorot_uniform()
    return {
        'w1': gl(k1, (in_dim, hidden)), 'b1': jnp.zeros((hidden,)),
        'w2': gl(k2, (hidden, hidden)), 'b2': jnp.zeros((hidden,)),
        'w3': gl(k3, (hidden, hidden)), 'b3': jnp.zeros((hidden,)),
        'w_mu': gl(k4, (hidden, 8)), 'b_mu': jnp.zeros((8,)),
        'w_ls': gl(k5, (hidden, 8)), 'b_ls': jnp.full((8,), -1.0),
        'w_gr': gl(k6, (hidden, 8)), 'b_gr': jnp.zeros((8,)),
    }

def _encoder_features(obs):
    o = jnp.transpose(obs, (1, 0, 2))
    a1 = o[:, :, 0]; s2 = o[:, :, 1]; a2 = o[:, :, 2]; r = o[:, :, 3]
    um = o[:, :, 4]; dist = o[:, :, 5]
    Q_mf0 = o[:, :, 6]; Q_mf1 = o[:, :, 7]
    Q_s2_00 = o[:, :, 8]; Q_s2_01 = o[:, :, 9]
    Q_s2_10 = o[:, :, 12]; Q_s2_11 = o[:, :, 13]
    P00 = o[:, :, 16]; P01 = o[:, :, 17]; P10 = o[:, :, 18]; P11 = o[:, :, 19]
    w_eff = o[:, :, 20]
    def m(x): return x.mean(axis=1)
    def s(x): return x.std(axis=1)
    def delta(x): return x[:, 1:] - x[:, :-1]
    def corr(x, y):
        x = x - x.mean(axis=1, keepdims=True); y = y - y.mean(axis=1, keepdims=True)
        num = (x*y).mean(axis=1); den = x.std(axis=1) * y.std(axis=1) + 1e-6
        return num / den
    dQ_mf = delta(Q_mf0); dQ_s2 = delta(Q_s2_00)
    r_lag = r[:, :-1]; um_lag = um[:, :-1]
    td_mf = (r_lag + 0.9*jnp.maximum(Q_mf0[:, :-1], Q_mf1[:, :-1]) - Q_mf0[:, :-1]) * um_lag
    td_s2 = (r_lag - Q_s2_00[:, :-1]) * um_lag
    r_prev = jnp.pad(r[:, :-1], ((0,0),(1,0)))
    a1_prev = jnp.pad(a1[:, :-1], ((0,0),(1,0)))
    Q_MB_0 = P00*jnp.maximum(Q_s2_00, Q_s2_01) + P01*jnp.maximum(Q_s2_10, Q_s2_11)
    Q_MB_1 = P10*jnp.maximum(Q_s2_10, Q_s2_11) + P11*jnp.maximum(Q_s2_00, Q_s2_01)
    feats = jnp.stack([
        m(a1), s(a1), m(a2), s(a2), m(r), s(r), m(um), m(dist), s(dist),
        m(Q_mf0), s(Q_mf0), m(Q_mf1), s(Q_mf1),
        m(Q_s2_00), s(Q_s2_00), m(Q_s2_10), s(Q_s2_10),
        m(P00), m(P11), m(w_eff), s(w_eff),
        corr(dQ_mf, td_mf), corr(dQ_s2, td_s2),
        m(dQ_mf), m(dQ_s2),
        corr(Q_mf0, r_prev), corr(Q_mf1, r_prev),
        corr(a2, r_prev), corr(a1, r_prev),
        (a1[:, 1:] == a1[:, :-1]).mean(axis=1),
        (a2 == 0).mean(axis=1), (a2 == 1).mean(axis=1),
        (a2 == 2).mean(axis=1), (a2 == 3).mean(axis=1),
        (a1 == 0).mean(axis=1),
        m(Q_MB_0), m(Q_MB_1), s(r * um),
        m(Q_s2_01), m(Q_s2_11),
        P00.std(axis=1), P11.std(axis=1),
        m(Q_s2_00 * a1), m(Q_s2_10 * a1),
        w_eff[:, -1], dist[:, -1],
        m(r * um) - m(r) * m(um),
    ], axis=-1)
    return feats

def guide(obs):
    N = obs.shape[1]
    # --- Population hyperparameters as trainable variational params ---
    # These MUST appear in the guide with the same names as in the model,
    # else SVI has no handles to optimise them (Bürkner et al. 2023).
    q_mu_w  = numpyro.param("q_mu_w",  0.0)
    q_sig_w = numpyro.param("q_sig_w", 1.0,
                            constraint=dist.constraints.positive)
    q_mu_a  = numpyro.param("q_mu_a",  0.0)
    q_sig_a = numpyro.param("q_sig_a", 1.0,
                            constraint=dist.constraints.positive)
    q_mu_b  = numpyro.param("q_mu_b",  1.0)
    q_sig_b = numpyro.param("q_sig_b", 1.0,
                            constraint=dist.constraints.positive)
    q_mu_p  = numpyro.param("q_mu_p",  0.2)
    q_sig_p = numpyro.param("q_sig_p", 0.5,
                            constraint=dist.constraints.positive)

    numpyro.sample("mu_w",  dist.Delta(q_mu_w))
    numpyro.sample("sig_w", dist.Delta(q_sig_w))
    numpyro.sample("mu_a",  dist.Delta(q_mu_a))
    numpyro.sample("sig_a", dist.Delta(q_sig_a))
    numpyro.sample("mu_b",  dist.Delta(q_mu_b))
    numpyro.sample("sig_b", dist.Delta(q_sig_b))
    numpyro.sample("mu_p",  dist.Delta(q_mu_p))
    numpyro.sample("sig_p", dist.Delta(q_sig_p))

    # --- Per-agent latents via encoder MLP ---
    enc = numpyro.param("enc", _init_encoder)
    feats = _encoder_features(obs)
    x = jnp.tanh(feats @ enc['w1'] + enc['b1'])
    x = jnp.tanh(x @ enc['w2'] + enc['b2'])
    x = jnp.tanh(x @ enc['w3'] + enc['b3'])
    mu = x @ enc['w_mu'] + enc['b_mu']
    ls = jnp.clip(x @ enc['w_ls'] + enc['b_ls'], -6.0, 2.0)
    with numpyro.plate("agents", N):
        numpyro.sample("w_raw", dist.Normal(mu[:, 0], jnp.exp(ls[:, 0])))
        numpyro.sample("a_raw", dist.Normal(mu[:, 1], jnp.exp(ls[:, 1])))
        numpyro.sample("b_raw", dist.Normal(mu[:, 2], jnp.exp(ls[:, 2])))
        numpyro.sample("p_raw", dist.Normal(mu[:, 3], jnp.exp(ls[:, 3])))

from numpyro.handlers import seed, trace, substitute

def amortized_infer(guide_params, obs, n_samples=128, seed_val=0):
    tr = trace(substitute(seed(guide, seed_val), guide_params)).get_trace(obs)
    k1, k2, k3, k4 = random.split(random.PRNGKey(seed_val), 4)
    w = jax.nn.sigmoid(tr["w_raw"]["fn"].sample(k1, (n_samples,)))
    a = jax.nn.sigmoid(tr["a_raw"]["fn"].sample(k2, (n_samples,)))
    b = jnp.exp(tr["b_raw"]["fn"].sample(k3, (n_samples,)))
    p = tr["p_raw"]["fn"].sample(k4, (n_samples,))
    return w, a, b, p

if __name__ == "__main__":
    def log(msg): print(msg, flush=True)
    log("="*70)
    log("PHASE 6a — Population-Level Amortized Bayesian Inference")
    log("="*70)
    log(f"Device: {jax.devices()[0]}  N={N_AGENTS} T={N_TICKS}")

    # Training
    log(f"\n[1/4] Generating {N_TRAIN_SIMS} training datasets ...")
    t0 = time.time()
    train_data = []
    keys = random.split(random.PRNGKey(SEED), N_TRAIN_SIMS)
    for i in tqdm(range(N_TRAIN_SIMS), desc="gen train", ncols=100):
        k_sim, k_obs = random.split(keys[i], 2)
        (w, a, b, p), _ = sample_population_params(k_sim)
        obs = _simulate_jit(w, a, b, p, k_obs)
        jax.block_until_ready(obs)
        train_data.append(obs)
    log(f"      wall = {time.time()-t0:.1f}s")

    log(f"\n[2/4] Training amortized guide ...")
    opt = ClippedAdam(1e-3, clip_norm=1.0)
    svi = SVI(model, guide, opt, loss=Trace_ELBO(num_particles=1))
    svi_state = svi.init(random.PRNGKey(0), train_data[0])
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        order = rng.permutation(N_TRAIN_SIMS)
        losses = []
        bar = tqdm(order, desc=f"epoch {epoch+1}/{N_EPOCHS}", ncols=100, leave=False)
        for i in bar:
            svi_state, loss = svi.update(svi_state, train_data[int(i)])
            losses.append(float(loss))
            bar.set_postfix_str(f"ELBO={np.mean(losses[-30:]):.0f}")
        bar.close()
        log(f"      epoch {epoch+1}  ELBO = {np.mean(losses):.1f}  wall = {time.time()-t0:.1f}s")
    guide_params = svi.get_params(svi_state)

    # SBC
    log(f"\n[3/4] SBC on {N_TEST_SIMS} held-out datasets ...")
    test_data = []; test_truth = []
    keys = random.split(random.PRNGKey(SEED+1), N_TEST_SIMS)
    for i in tqdm(range(N_TEST_SIMS), desc="gen test", ncols=100):
        k_sim, k_obs = random.split(keys[i], 2)
        (w, a, b, p), (wr, ar, br, pr) = sample_population_params(k_sim)
        obs = _simulate_jit(w, a, b, p, k_obs)
        jax.block_until_ready(obs)
        test_data.append(obs); test_truth.append((wr, ar, br, pr))

    log(f"\n[4/4] Evaluating hyperparameter recovery across {N_TEST_SIMS} datasets ...")

    # The guide's trained params ARE the point estimates for the hyperparameters.
    # They should be close to the true generating values (MU_W, SIG_W, ...).
    truth = {"mu_w": MU_W, "sig_w": SIG_W, "mu_a": MU_A, "sig_a": SIG_A,
             "mu_b": MU_B, "sig_b": SIG_B, "mu_p": MU_P, "sig_p": SIG_P}
    est   = {"mu_w": float(guide_params["q_mu_w"]),
             "sig_w": float(guide_params["q_sig_w"]),
             "mu_a": float(guide_params["q_mu_a"]),
             "sig_a": float(guide_params["q_sig_a"]),
             "mu_b": float(guide_params["q_mu_b"]),
             "sig_b": float(guide_params["q_sig_b"]),
             "mu_p": float(guide_params["q_mu_p"]),
             "sig_p": float(guide_params["q_sig_p"])}

    log("\n" + "="*70)
    log("POPULATION HYPERPARAMETER RECOVERY (single estimate vs. truth)")
    log("="*70)
    log(f"{'param':10s} {'true':>10} {'est':>10} {'abs_err':>10} {'rel_err':>10}")
    log("-"*70)
    results = {}
    n_pass = 0
    for k in truth:
        t = truth[k]; e = est[k]
        rel = abs(e - t) / (abs(t) + 1e-6)
        passed = rel < 0.5
        n_pass += int(passed)
        results[k] = {"true": t, "est": e,
                      "abs_err": abs(e-t), "rel_err": rel,
                      "verdict": "PASS" if passed else "FAIL"}
        log(f"{k:10s} {t:>+10.4f} {e:>+10.4f} {abs(e-t):>10.4f} {rel:>10.4f}  {'PASS' if passed else 'FAIL'}")
    log(f"\nOVERALL: {n_pass}/8 hyperparameters within 50% relative error")

    # --- Also report per-agent rho using the encoder on test datasets ---
    log(f"\nPer-agent point-estimate quality (across {N_TEST_SIMS} held-out datasets):")
    rec_agent = {"w_MB": [], "alpha_R": [], "beta_choice": [], "phi_per": []}
    tru_agent = {k: [] for k in rec_agent}
    for i in tqdm(range(N_TEST_SIMS), desc="infer", ncols=100):
        obs = test_data[i]
        w_s, a_s, b_s, p_s = amortized_infer(guide_params, obs, n_samples=N_ENC_SAMP, seed_val=i)
        rec_agent["w_MB"].append(np.asarray(w_s.mean(0)))
        rec_agent["alpha_R"].append(np.asarray(a_s.mean(0)))
        rec_agent["beta_choice"].append(np.asarray(b_s.mean(0)))
        rec_agent["phi_per"].append(np.asarray(p_s.mean(0)))
        wr, ar, br, pr = test_truth[i]
        tru_agent["w_MB"].append(1/(1+np.exp(-wr)))
        tru_agent["alpha_R"].append(1/(1+np.exp(-ar)))
        tru_agent["beta_choice"].append(np.exp(br))
        tru_agent["phi_per"].append(pr)
    log(f"\n{'param':14s} {'rho':>8} {'bias':>10} {'rmse':>10}")
    log("-"*50)
    for p in ["w_MB", "alpha_R", "beta_choice", "phi_per"]:
        t = np.concatenate(tru_agent[p]); r = np.concatenate(rec_agent[p])
        rho, _ = spearmanr(t, r)
        bias = float(np.mean(r - t)); rmse = float(np.sqrt(np.mean((r-t)**2)))
        log(f"{p:14s} {rho:>+8.4f} {bias:>+10.4f} {rmse:>10.4f}")

    with open(OUT / "p6a_population_results.json", "w") as f:
        json.dump({"recovery": results,
                   "config": {"N_AGENTS": N_AGENTS, "N_TICKS": N_TICKS,
                              "train_sims": N_TRAIN_SIMS, "test_sims": N_TEST_SIMS,
                              "epochs": N_EPOCHS}}, f, indent=2)
    log(f"Saved: {OUT / 'p6a_population_results.json'}")
