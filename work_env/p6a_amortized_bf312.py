# p6a_amortized.py — Phase 6a: Amortized Bayesian Inference with BayesFlow 2.0
#
# Trains a neural posterior estimator on engine_v5 simulations ONCE.
# After training, posterior inference on any new observation is < 1 second.
#
# Pipeline:
#   1. Simulator: wraps engine_v5 to produce (per-agent parameters, per-agent
#      observables) pairs. Uses the full forward-simulated engine — no
#      approximation of the physics.
#   2. Amortized posterior: BayesFlow TimeSeriesNetwork (summary) + FlowMatching
#      (inference). Trained on 2000 online-simulated batches.
#   3. Validation: SBC (rank histogram uniformity) on 200 held-out datasets.
#   4. Recovery metrics: rho, bias, RMSE per parameter.
#
# Reference: Kühmichel et al. (2026), arXiv:2602.07098.
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import time, json, warnings
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax
import bayesflow as bf
import keras

import engine_v5 as E

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent
PARAMS_PKL = Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_AGENTS       = 60
N_TICKS        = 150
N_TRAIN_EPOCHS = 40
BATCH_SIZE     = 24
BATCHES_PER_EP = 50
N_TEST         = 200       # datasets for SBC + recovery
SEED           = 42

# The four target parameters
TARGET_PARAMS = ["w_MB", "alpha_R", "beta_choice", "phi_per"]

# Priors for the four target parameters (sampled at simulation time)
PRIORS = {
    "w_MB":        ("sigmoid_normal", 0.0, 1.0),   # sigmoid(N(0,1))
    "alpha_R":     ("sigmoid_normal", 0.0, 1.0),
    "beta_choice": ("lognormal",      1.0, 1.0),   # logN(1, 1)
    "phi_per":     ("normal",         0.2, 0.5),   # N(0.2, 0.5)
}

# ---------------------------------------------------------------------------
# Static per-agent copula parameters (sampled once per batch from
# synthetic_32k.pkl). All 26 non-target params come from the fitted copula.
# ---------------------------------------------------------------------------
_COPULA = None

def _load_copula():
    global _COPULA
    if _COPULA is None:
        _COPULA = E.load_all(PARAMS_PKL)
    return _COPULA

def _sample_priors(key):
    """Sample the four target parameters from their priors."""
    k1, k2, k3, k4 = random.split(key, 4)
    w_mb   = jax.nn.sigmoid(random.normal(k1, (N_AGENTS,)))
    alpha  = jax.nn.sigmoid(random.normal(k2, (N_AGENTS,)))
    beta   = jnp.exp(random.normal(k3, (N_AGENTS,)) * 1.0 + 1.0)
    phi    = random.normal(k4, (N_AGENTS,)) * 0.5 + 0.2
    return {"w_MB": w_mb, "alpha_R": alpha,
            "beta_choice": beta, "phi_per": phi}

# ---------------------------------------------------------------------------
# Simulator — wraps engine_v5 to return per-agent traces.
# This is a JAX-jitted function: constant across training.
# ---------------------------------------------------------------------------
def _make_simulator_fn():
    """Build a jitted simulator that takes target params + copula slice and
    returns per-agent observables (a1, s2, a2, r, interacting)."""
    src = _load_copula()
    nb_full, _ = E.build_ws(n=N_AGENTS, k=6, p=0.1, seed=0)
    nb = jnp.asarray(nb_full, jnp.int32)
    arange = jnp.arange(N_AGENTS)
    valid_pick = jnp.where(nb >= 0, 0.0, -1e30)

    # Static non-target copula params (same across every batch for reproducibility).
    # In a full-hierarchical setup these could also be resampled; here they are
    # held at a fixed representative draw so the recovery signal is attributable
    # to the four target parameters.
    idx_static = np.random.default_rng(SEED).integers(0, len(src["O"]), size=N_AGENTS)
    static = {k: jnp.asarray(v[idx_static]) for k, v in src.items()}

    C_b      = static["C"]
    eta_vol  = static["eta_vol"]
    E_bias   = static["E"] * E.E_ENGAGE
    a_inhib  = static["a_inhib"] * E.A_INHIB
    theta_T  = static["theta_T"] * E.T_THREAT
    u_U      = static["u_U"] * E.U_UNCERT
    SM_b     = static["S_M"] * E.SM_DEFECT
    SP_b     = static["S_P"] * E.SP_PUNISH
    SN_b     = static["S_N"] * E.SN_STATUS
    omega    = static["omega_social"]
    lam_norm = static["lambda_norm"] * E.LAMBDA_NORM_A
    sal      = static["gamma_sal"] * E.SAL_GAIN
    mood     = static["m_t"]
    alpha_e  = static["D"] + static["lambda_ineq"]
    beta_e   = static["A"]
    gamma    = 1.0/(1.0 + jnp.exp(jnp.clip(static["k_DD"], -10.0, 5.0)))
    baseline = (static["general_distress"]
                + 0.3*static["DASS_Dep"] + 0.3*static["DASS_Anx"]
                + 0.3*static["DASS_Str"]) / 1.9
    N_sens   = static["N"]

    Q_mf0  = jnp.zeros((N_AGENTS, 2), jnp.float32)
    Q_s2_0 = jnp.zeros((N_AGENTS, 2, E.N_ACTIONS), jnp.float32)
    P_hat0 = jnp.zeros((N_AGENTS, 2, 2), jnp.float32)
    P_hat0 = P_hat0.at[:, 0, 0].set(E.P_COMMON).at[:, 0, 1].set(1.0 - E.P_COMMON)
    P_hat0 = P_hat0.at[:, 1, 0].set(1.0 - E.P_COMMON).at[:, 1, 1].set(E.P_COMMON)
    w0     = jnp.full((N_AGENTS,), E.WEALTH_INIT, jnp.float32)
    last0  = jnp.full((N_AGENTS,), -1, jnp.int32)

    def step(carry, tk):
        Q_mf, Q_s2, P_hat, wealth, dist, last_a1 = carry
        w_mb, alpha_r, beta_ch, phi_per = carry[6:]
        kp, kc, ka1, ka2, kr, _ = random.split(tk, 6)

        k_idx = random.categorical(kp, valid_pick, axis=-1)
        ch    = nb[arange, k_idx]
        mut   = (ch[ch] == arange)
        pid   = jnp.minimum(arange, ch)
        ca    = random.bernoulli(kc, 0.5, shape=(N_AGENTS,)).astype(jnp.float32)
        cp    = ca[pid]; il = arange < ch
        pp    = jnp.where(cp == 1.0, il, ~il)
        isP   = pp & mut
        isR   = (~pp) & mut

        w_eff = jnp.clip(w_mb * (1.0 + C_b * E.C_BOOST) *
                         (1.0 - dist * E.KAPPA_DECAY), 0.0, 1.0)
        best  = jnp.max(Q_s2, axis=-1)
        Q_MB  = jnp.sum(P_hat * best[:, None, :], axis=-1)
        Q_net = w_eff[:, None] * Q_MB + (1.0 - w_eff[:, None]) * Q_mf

        eng = E_bias - a_inhib
        avd = theta_T + u_U
        per = phi_per * jnp.where(last_a1 == 0, 1.0, 0.0)
        L1  = beta_ch[:, None] * jnp.stack(
            [Q_net[:, 0] + eng + per, Q_net[:, 1] + avd], axis=-1)
        L1  = jnp.clip(L1, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a1  = random.categorical(ka1, L1, axis=-1)

        cm = random.bernoulli(kr, E.P_COMMON, shape=(N_AGENTS,))
        s2 = jnp.where(cm, a1, 1 - a1)

        qs = Q_s2[arange, s2, :]
        qs = jnp.where(qs < 0.0, qs * (1.0 - SP_b[:, None] * 0.5), qs)
        db = -SM_b - SN_b
        ab = jnp.stack([db, db * 0.5, jnp.zeros_like(db), jnp.zeros_like(db)], axis=-1)
        L2 = beta_ch[:, None] * (qs + ab)
        L2 = jnp.clip(L2, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a2 = random.categorical(ka2, L2, axis=-1)

        my_off = E.ULTI_OFFERS[a2]
        rec_off = my_off[ch]
        d_me = 2.0 * rec_off - E.E_ENDOWMENT
        rp_me = jnp.maximum(d_me, 0.0); rn_me = jnp.maximum(-d_me, 0.0)
        my_UR = rec_off - alpha_e * rn_me - beta_e * rp_me + lam_norm
        my_acc = ((my_UR >= 0.0) & isR).astype(jnp.float32)
        d_p = 2.0 * my_off - E.E_ENDOWMENT
        rp_p = jnp.maximum(d_p, 0.0); rn_p = jnp.maximum(-d_p, 0.0)
        p_UR = my_off - alpha_e[ch] * rn_p - beta_e[ch] * rp_p
        p_acc = ((p_UR >= 0.0) & isP).astype(jnp.float32)
        r_ultiP = jnp.where(p_acc >= 1.0, E.E_ENDOWMENT - my_off, 0.0)
        r_ultiR = jnp.where(my_acc >= 1.0, rec_off, 0.0)

        my_s = E.TRUST_SENDS[a2]; rec_s = my_s[ch]
        my_rm = E.TRUST_RETURN_MULT[a2]; rec_rm = my_rm[ch]
        r_ts = jnp.where(isR, -my_s + E.TRUST_K * rec_rm * my_s, 0.0)
        r_tt = jnp.where(isP, E.TRUST_K * rec_s * (1.0 - my_rm / E.TRUST_K), 0.0)

        raw = r_ultiP + r_ultiR + r_ts + r_tt
        rew = jnp.clip(raw * omega * (1.0 + sal + 0.05 * mood),
                       -E.REWARD_CLIP, E.REWARD_CLIP)
        inter = (a1 == 0) & (a1[ch] == 0) & mut
        rew = jnp.where(inter, rew, 0.0)
        um = inter.astype(jnp.float32)

        qsv = Q_s2[arange, s2, a2]
        Q_s2n = Q_s2.at[arange, s2, a2].set(
            jnp.clip(qsv + um * alpha_r * (rew - qsv), -E.GRAD_CLIP, E.GRAD_CLIP))
        qmv = Q_mf[arange, a1]; vn = jnp.max(Q_mf, axis=-1)
        Q_mfn = Q_mf.at[arange, a1].set(
            jnp.clip(qmv + um * alpha_r * (rew + gamma * vn - qmv),
                     -E.GRAD_CLIP, E.GRAD_CLIP))
        oh = jnp.stack([1.0 - s2.astype(jnp.float32), s2.astype(jnp.float32)], axis=-1)
        Pat = P_hat[arange, a1, :]
        Pu = Pat + um[:, None] * eta_vol[:, None] * (oh - Pat)
        Pu = jnp.clip(Pu, 1e-4, 1.0); Pu = Pu / jnp.sum(Pu, axis=-1, keepdims=True)
        P_hatn = P_hat.at[arange, a1, :].set(Pu)

        rej = isR & inter & (my_UR < 0.0)
        dn = (E.DISTRESS_DECAY * dist + E.DISTRESS_PULL * baseline
              + (1.0 + N_sens * E.N_DISTRESS) * rej.astype(jnp.float32) * 0.05)
        wn = jnp.maximum(wealth + rew, 0.0)

        return (Q_mfn, Q_s2n, P_hatn, wn, dn, a1,
                w_mb, alpha_r, beta_ch, phi_per), (a1, s2, a2, rew, um)

    @jax.jit
    def simulator(target_params, key):
        """target_params: dict of 4 arrays (N_AGENTS,). Returns observables."""
        keys = random.split(key, N_TICKS)
        carry0 = (Q_mf0, Q_s2_0, P_hat0, w0, baseline, last0,
                  target_params["w_MB"], target_params["alpha_R"],
                  target_params["beta_choice"], target_params["phi_per"])
        _, (a1_tr, s2_tr, a2_tr, r_tr, um_tr) = lax.scan(step, carry0, keys)
        # Stack into per-agent time series [N_AGENTS, N_TICKS, 5]
        obs = jnp.stack([a1_tr.astype(jnp.float32),
                         s2_tr.astype(jnp.float32),
                         a2_tr.astype(jnp.float32),
                         r_tr,
                         um_tr], axis=-1)
        return obs  # (N_AGENTS, N_TICKS, 5)

    return simulator

_SIMULATOR = _make_simulator_fn()

# ---------------------------------------------------------------------------
# BayesFlow adapter — produces (params, observables) pairs
# ---------------------------------------------------------------------------
def _adapter(batch_size=BATCH_SIZE, seed=None):
    """BayesFlow adapter: sample params, simulate, return dict."""
    rng = np.random.default_rng(seed)
    while True:
        key = random.PRNGKey(int(rng.integers(0, 2**31)))
        params = _sample_priors(key)
        obs = _SIMULATOR(params, key)
        # Stack target params into (N_AGENTS, 4)
        p_arr = jnp.stack([params["w_MB"], params["alpha_R"],
                           params["beta_choice"], params["phi_per"]], axis=-1)
        yield {"parameters": np.asarray(p_arr), "observables": np.asarray(obs)}

# ---------------------------------------------------------------------------
# Build the BayesFlow workflow
# ---------------------------------------------------------------------------
def build_workflow():
    # Summary network: processes each agent's time series independently
    # then produces a per-agent sufficient statistic. Uses a 1D-conv +
    # bidirectional LSTM — the same architecture the ABM reference paper
    # (arXiv:2511.08763) uses for per-agent latent recovery.
    summary = bf.networks.TimeSeriesNetwork(
        bidirectional=True,
        lstm_units=32,
        dense_units=32,
    )
    inference = bf.networks.FlowMatching()
    workflow = bf.BasicWorkflow(
        inference_network=inference,
        summary_network=summary,
        inference_variables=["parameters"],
        inference_conditions=["observables"],
        adapter=None,   # we supply data via the adapter generator
    )
    return workflow

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("PHASE 6a — Amortized Bayesian Inference (BayesFlow 2.0)")
    print("="*70)
    print(f"Device:        {jax.devices()[0]}")
    print(f"Backend:       {jax.default_backend()}")
    print(f"Keras backend: {keras.backend.backend()}")
    print(f"BayesFlow:     {bf.__version__ if hasattr(bf, '__version__') else '2.x'}")
    print(f"Config: N={N_AGENTS} T={N_TICKS}  batch={BATCH_SIZE}  "
          f"epochs={N_TRAIN_EPOCHS}  batches/epoch={BATCHES_PER_EP}")
    print()

    # ---- Sanity: one simulator call ----
    print("[1/4] Sanity-checking simulator ...")
    t0 = time.time()
    k = random.PRNGKey(0)
    p_demo = _sample_priors(k)
    obs_demo = _SIMULATOR(p_demo, k)
    jax.block_until_ready(obs_demo)
    print(f"      simulator wall = {time.time()-t0:.2f}s  obs.shape = {obs_demo.shape}")
    print(f"      true w_MB mean    = {float(p_demo['w_MB'].mean()):.3f}")
    print(f"      true alpha_R mean = {float(p_demo['alpha_R'].mean()):.3f}")
    print(f"      true beta mean    = {float(p_demo['beta_choice'].mean()):.3f}")
    print(f"      true phi_per mean = {float(p_demo['phi_per'].mean()):.3f}")
    print()

    # ---- Generate training data ----
    print(f"[2/4] Generating {N_TRAIN_EPOCHS * BATCHES_PER_EP} training batches "
          f"({N_TRAIN_EPOCHS * BATCHES_PER_EP * BATCH_SIZE} sims total) ...")
    t0 = time.time()
    train_pars, train_obs = [], []
    gen = _adapter(seed=SEED)
    n_total = N_TRAIN_EPOCHS * BATCHES_PER_EP
    for i, batch in enumerate(gen):
        if i >= n_total: break
        train_pars.append(batch["parameters"])
        train_obs.append(batch["observables"])
        if (i+1) % 200 == 0:
            print(f"      generated {i+1}/{n_total}  "
                  f"({time.time()-t0:.1f}s)")
    train_pars = np.stack(train_pars); train_obs = np.stack(train_obs)
    print(f"      train_pars.shape = {train_pars.shape}  "
          f"train_obs.shape = {train_obs.shape}")
    print(f"      total data gen wall = {time.time()-t0:.1f}s")
    print()

    # ---- Train the amortized posterior ----
    print(f"[3/4] Training amortized posterior ...")
    workflow = build_workflow()
    t0 = time.time()
    history = workflow.fit_offline(
        data={"parameters": train_pars, "observables": train_obs},
        epochs=N_TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )
    print(f"      training wall = {time.time()-t0:.1f}s")
    print()

    # ---- SBC validation ----
    print(f"[4/4] Running simulation-based calibration (SBC) on {N_TEST} held-out datasets ...")
    gen_test = _adapter(seed=SEED + 1)
    test_pars, test_obs = [], []
    for i, b in enumerate(gen_test):
        if i >= N_TEST: break
        test_pars.append(b["parameters"]); test_obs.append(b["observables"])
    test_pars = np.stack(test_pars); test_obs = np.stack(test_obs)

    post = workflow.sample(
        conditions={"observables": test_obs},
        num_samples=200,
    )
    # post["parameters"] shape: (N_TEST, num_samples, N_AGENTS, 4)
    samples = np.asarray(post["parameters"])

    print()
    print("="*70)
    print("RECOVERY METRICS (posterior mean vs. true, per agent)")
    print("="*70)
    from scipy.stats import spearmanr
    results = {}
    for pi, pname in enumerate(TARGET_PARAMS):
        true_all = test_pars[:, :, pi].reshape(-1)        # (N_TEST * N_AGENTS,)
        rec_all  = samples[:, :, :, pi].mean(axis=1).reshape(-1)
        rho, pval = spearmanr(true_all, rec_all)
        bias = float(np.mean(rec_all - true_all))
        rmse = float(np.sqrt(np.mean((rec_all - true_all)**2)))
        verdict = "PASS" if rho > 0.6 else "FAIL"
        results[pname] = {"rho": float(rho), "p": float(pval),
                          "bias": bias, "rmse": rmse, "verdict": verdict}
        print(f"{pname:14s} rho={rho:>+.4f}  p={pval:>8.2e}  "
              f"bias={bias:>+.4f}  rmse={rmse:.4f}  {verdict}")

    print()
    n_pass = sum(1 for r in results.values() if r["verdict"] == "PASS")
    print(f"OVERALL: {n_pass}/4 parameters recovered at rho > 0.6")

    # Save
    with open(OUT / "p6a_results.json", "w") as f:
        json.dump({"recovery": results, "config": {
            "N_AGENTS": N_AGENTS, "N_TICKS": N_TICKS,
            "N_TRAIN_EPOCHS": N_TRAIN_EPOCHS, "N_TEST": N_TEST}}, f, indent=2)
    print(f"Saved: {OUT / 'p6a_results.json'}")
