# hba_fitter.py — Full-rigor Hierarchical Bayesian Analysis (HBA) fitter
#
# Architecture:
#   - Forward-simulate engine_v5 to generate a ground-truth observation dataset.
#   - Define a NumPyro model that re-simulates the exact Q-table dynamics
#     conditioned on the OBSERVED (a1, s2, a2, r) trajectory.
#   - The model has 4 individual-level parameters (w_MB, alpha_R, beta_choice,
#     phi_per) sampled hierarchically via non-centered parameterization.
#   - All other parameters (26) are held fixed from the copula.
#   - Likelihood is registered as `numpyro.factor` over the sum of per-tick
#     log P(a1_obs | logits_a1) + log P(a2_obs | logits_a2).
#   - Because s2_obs and r_obs are observed, Q-table updates are deterministic
#     given parameters. NUTS computes exact gradients through the scan.
#
# Performance: jax.lax.scan + JIT; vectorized over agents; GPU-accelerated.
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
from numpyro.infer import MCMC, NUTS
from scipy.stats import spearmanr
from jax import random, lax

import engine_v5 as E

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
N_AGENTS       = 300
N_TICKS        = 150
N_WARMUP       = 500
N_SAMPLES      = 1500
N_CHAINS       = 2
SEED           = 42
RECOVER_PARAMS = ["w_MB", "alpha_R", "beta_choice", "phi_per"]
OUT = Path(__file__).parent

# -----------------------------------------------------------------------------
# 1. Simulate observation dataset from engine_v5
# -----------------------------------------------------------------------------
def simulate_observations(n_agents, n_ticks, seed):
    """Run the coupled engine for n_agents x n_ticks; return per-agent
    (a1, s2, a2, r) traces and the true per-agent parameters."""
    src = E.load_all(Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl")
    idx = np.random.default_rng(seed).integers(0, len(src["O"]), size=n_agents)
    P = {k: jnp.asarray(v[idx]) for k, v in src.items()}
    nb, _ = E.build_ws(n=n_agents, k=6, p=0.1, seed=0)
    neighbors = jnp.asarray(nb, jnp.int32)

    arange_N = jnp.arange(n_agents)
    valid_pick = jnp.where(neighbors >= 0, 0.0, -1e30)

    # per-agent static quantities (from copula, held fixed)
    C_b       = P["C"]
    eta_vol   = P["eta_vol"]
    E_bias    = P["E"] * E.E_ENGAGE
    a_inhib   = P["a_inhib"] * E.A_INHIB
    theta_T   = P["theta_T"] * E.T_THREAT
    u_U       = P["u_U"] * E.U_UNCERT
    SM_b      = P["S_M"] * E.SM_DEFECT
    SP_b      = P["S_P"] * E.SP_PUNISH
    SN_b      = P["S_N"] * E.SN_STATUS
    omega_soc = P["omega_social"]
    lam_norm  = P["lambda_norm"] * E.LAMBDA_NORM_A
    sal       = P["gamma_sal"] * E.SAL_GAIN
    mood      = P["m_t"]
    alpha_env = P["D"] + P["lambda_ineq"]
    beta_env  = P["A"]
    gamma     = 1.0 / (1.0 + jnp.exp(jnp.clip(P["k_DD"], -10.0, 5.0)))
    baseline  = (P["general_distress"] + 0.3*P["DASS_Dep"]
                 + 0.3*P["DASS_Anx"] + 0.3*P["DASS_Str"]) / 1.9
    lr        = P["alpha_R"]
    beta_soft = P["beta_choice"]
    phi       = P["phi_per"]
    w_mb_base = P["w_MB"]
    N_sens    = P["N"]

    # initial state
    Q_mf0   = jnp.zeros((n_agents, 2), jnp.float32)
    Q_s2_0  = jnp.zeros((n_agents, 2, E.N_ACTIONS), jnp.float32)
    P_hat0  = jnp.zeros((n_agents, 2, 2), jnp.float32)
    P_hat0  = P_hat0.at[:, 0, 0].set(E.P_COMMON).at[:, 0, 1].set(1.0 - E.P_COMMON)
    P_hat0  = P_hat0.at[:, 1, 0].set(1.0 - E.P_COMMON).at[:, 1, 1].set(E.P_COMMON)
    wealth0 = jnp.full((n_agents,), E.WEALTH_INIT, jnp.float32)
    last0   = jnp.full((n_agents,), -1, jnp.int32)

    keys = random.split(random.PRNGKey(seed), n_ticks)

    def step(carry, tick_key):
        Q_mf, Q_s2, P_hat, wealth, distress, last_a1 = carry
        k_pick, k_coin, k_a1, k_a2, k_role, _ = random.split(tick_key, 6)

        # matching
        k_idx = random.categorical(k_pick, valid_pick, axis=-1)
        chosen = neighbors[arange_N, k_idx]
        is_mutual = (chosen[chosen] == arange_N)
        pair_id = jnp.minimum(arange_N, chosen)
        coin_all = random.bernoulli(k_coin, 0.5, shape=(n_agents,)).astype(jnp.float32)
        coin_pair = coin_all[pair_id]
        is_lower = arange_N < chosen
        is_prop_pre = jnp.where(coin_pair == 1.0, is_lower, ~is_lower)
        is_proposer = is_prop_pre & is_mutual
        is_responder = (~is_prop_pre) & is_mutual

        # arbitration
        w_mb_eff = jnp.clip(w_mb_base * (1.0 + C_b * E.C_BOOST)
                            * (1.0 - distress * E.KAPPA_DECAY), 0.0, 1.0)
        best_s2 = jnp.max(Q_s2, axis=-1)
        Q_MB = jnp.sum(P_hat * best_s2[:, None, :], axis=-1)
        Q_net = w_mb_eff[:, None] * Q_MB + (1.0 - w_mb_eff[:, None]) * Q_mf

        # stage-1
        engage_adv = E_bias - a_inhib
        avoid_adv  = theta_T + u_U
        persev = phi * jnp.where(last_a1 == 0, 1.0, 0.0)
        L1 = beta_soft[:, None] * jnp.stack(
            [Q_net[:, 0] + engage_adv + persev, Q_net[:, 1] + avoid_adv], axis=-1)
        L1 = jnp.clip(L1, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a1 = random.categorical(k_a1, L1, axis=-1)

        # transition
        common = random.bernoulli(k_role, E.P_COMMON, shape=(n_agents,))
        s2 = jnp.where(common, a1, 1 - a1)

        # stage-2
        q_s2 = Q_s2[arange_N, s2, :]
        q_s2 = jnp.where(q_s2 < 0.0, q_s2 * (1.0 - SP_b[:, None] * 0.5), q_s2)
        dark_bias = -SM_b - SN_b
        action_bonus = jnp.stack([dark_bias, dark_bias*0.5,
                                  jnp.zeros_like(dark_bias), jnp.zeros_like(dark_bias)], axis=-1)
        L2 = beta_soft[:, None] * (q_s2 + action_bonus)
        L2 = jnp.clip(L2, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a2 = random.categorical(k_a2, L2, axis=-1)

        # ultimatum
        my_offer = E.ULTI_OFFERS[a2]
        received_offer = my_offer[chosen]
        d_me = 2.0 * received_offer - E.E_ENDOWMENT
        rp_me = jnp.maximum(d_me, 0.0); rn_me = jnp.maximum(-d_me, 0.0)
        my_U_R = received_offer - alpha_env*rn_me - beta_env*rp_me + lam_norm
        my_accept = ((my_U_R >= 0.0) & is_responder).astype(jnp.float32)

        d_p = 2.0 * my_offer - E.E_ENDOWMENT
        rp_p = jnp.maximum(d_p, 0.0); rn_p = jnp.maximum(-d_p, 0.0)
        partner_U_R = my_offer - alpha_env[chosen]*rn_p - beta_env[chosen]*rp_p
        partner_accept = ((partner_U_R >= 0.0) & is_proposer).astype(jnp.float32)

        r_ulti_P = jnp.where(partner_accept >= 1.0, E.E_ENDOWMENT - my_offer, 0.0)
        r_ulti_R = jnp.where(my_accept >= 1.0, received_offer, 0.0)

        # trust
        is_trust_sender = is_responder; is_trust_trustee = is_proposer
        my_send = E.TRUST_SENDS[a2]; received_send = my_send[chosen]
        my_return_mult = E.TRUST_RETURN_MULT[a2]; received_return_mult = my_return_mult[chosen]
        r_trust_sender  = jnp.where(is_trust_sender,
                                    -my_send + E.TRUST_K * received_return_mult * my_send, 0.0)
        r_trust_trustee = jnp.where(is_trust_trustee,
                                    E.TRUST_K * received_send * (1.0 - my_return_mult / E.TRUST_K), 0.0)

        raw_reward = r_ulti_P + r_ulti_R + r_trust_sender + r_trust_trustee
        reward = jnp.clip(raw_reward * omega_soc * (1.0 + sal + 0.05 * mood),
                          -E.REWARD_CLIP, E.REWARD_CLIP)
        interacting = (a1 == 0) & (a1[chosen] == 0) & is_mutual
        reward = jnp.where(interacting, reward, 0.0)

        # Q updates
        update_mask = interacting.astype(jnp.float32)
        q_s2_prev = Q_s2[arange_N, s2, a2]
        q_s2_new_val = jnp.clip(q_s2_prev + update_mask*lr*(reward - q_s2_prev), -E.GRAD_CLIP, E.GRAD_CLIP)
        Q_s2_new = Q_s2.at[arange_N, s2, a2].set(q_s2_new_val)

        q_mf_prev = Q_mf[arange_N, a1]
        v_next = jnp.max(Q_mf, axis=-1)
        q_mf_new_val = jnp.clip(q_mf_prev + update_mask*lr*(reward + gamma*v_next - q_mf_prev),
                                -E.GRAD_CLIP, E.GRAD_CLIP)
        Q_mf_new = Q_mf.at[arange_N, a1].set(q_mf_new_val)

        one_hot = jnp.stack([1.0 - s2.astype(jnp.float32), s2.astype(jnp.float32)], axis=-1)
        P_at = P_hat[arange_N, a1, :]
        P_upd = P_at + update_mask[:, None] * eta_vol[:, None] * (one_hot - P_at)
        P_upd = jnp.clip(P_upd, 1e-4, 1.0); P_upd = P_upd / jnp.sum(P_upd, axis=-1, keepdims=True)
        P_hat_new = P_hat.at[arange_N, a1, :].set(P_upd)

        # distress
        rejected = is_responder & interacting & (my_U_R < 0.0)
        new_distress = (E.DISTRESS_DECAY * distress + E.DISTRESS_PULL * baseline
                        + (1.0 + N_sens * E.N_DISTRESS) * rejected.astype(jnp.float32) * 0.05)
        new_wealth = jnp.maximum(wealth + reward, 0.0)

        return (Q_mf_new, Q_s2_new, P_hat_new, new_wealth, new_distress, a1), (a1, s2, a2, reward)

    @jax.jit
    def run(carry, ks):
        return lax.scan(step, carry, ks)

    carry0 = (Q_mf0, Q_s2_0, P_hat0, wealth0, baseline, last0)
    _, (a1_tr, s2_tr, a2_tr, r_tr) = run(carry0, keys)
    jax.block_until_ready(a1_tr)

    return {
        "a1": np.asarray(a1_tr, dtype=np.int32),   # [N, T]
        "s2": np.asarray(s2_tr, dtype=np.int32),
        "a2": np.asarray(a2_tr, dtype=np.int32),
        "r":  np.asarray(r_tr,  dtype=np.float32),
        "P":  {k: jnp.asarray(v) for k, v in P.items()},
        "true": {p: np.asarray(P[p]) for p in RECOVER_PARAMS},
    }

# -----------------------------------------------------------------------------
# 2. NumPyro model
# -----------------------------------------------------------------------------
def build_model(P_fixed, n_agents, n_ticks):
    """Return a NumPyro model closure. All fixed quantities are constants."""
    # Static per-agent (fixed from copula)
    C_b       = P_fixed["C"]
    eta_vol   = P_fixed["eta_vol"]
    E_bias    = P_fixed["E"] * E.E_ENGAGE
    a_inhib   = P_fixed["a_inhib"] * E.A_INHIB
    theta_T   = P_fixed["theta_T"] * E.T_THREAT
    u_U       = P_fixed["u_U"] * E.U_UNCERT
    SM_b      = P_fixed["S_M"] * E.SM_DEFECT
    SP_b      = P_fixed["S_P"] * E.SP_PUNISH
    SN_b      = P_fixed["S_N"] * E.SN_STATUS
    gamma     = 1.0 / (1.0 + jnp.exp(jnp.clip(P_fixed["k_DD"], -10.0, 5.0)))
    baseline  = (P_fixed["general_distress"] + 0.3*P_fixed["DASS_Dep"]
                 + 0.3*P_fixed["DASS_Anx"] + 0.3*P_fixed["DASS_Str"]) / 1.9

    arange_N = jnp.arange(n_agents)

    Q_mf0  = jnp.zeros((n_agents, 2), jnp.float32)
    Q_s2_0 = jnp.zeros((n_agents, 2, E.N_ACTIONS), jnp.float32)
    P_hat0 = jnp.zeros((n_agents, 2, 2), jnp.float32)
    P_hat0 = P_hat0.at[:, 0, 0].set(E.P_COMMON).at[:, 0, 1].set(1.0 - E.P_COMMON)
    P_hat0 = P_hat0.at[:, 1, 0].set(1.0 - E.P_COMMON).at[:, 1, 1].set(E.P_COMMON)
    last0  = jnp.full((n_agents,), -1, jnp.int32)

    def model(a1_obs, s2_obs, a2_obs, r_obs):
        # ---- Hyperpriors ----
        mu_w    = numpyro.sample("mu_w",    dist.Normal(0.0, 1.0))
        sig_w   = numpyro.sample("sig_w",   dist.HalfNormal(1.0))
        mu_a    = numpyro.sample("mu_a",    dist.Normal(0.0, 1.0))
        sig_a   = numpyro.sample("sig_a",   dist.HalfNormal(1.0))
        mu_b    = numpyro.sample("mu_b",    dist.Normal(1.0, 1.0))
        sig_b   = numpyro.sample("sig_b",   dist.HalfNormal(1.0))
        mu_p    = numpyro.sample("mu_p",    dist.Normal(0.0, 1.0))
        sig_p   = numpyro.sample("sig_p",   dist.HalfNormal(1.0))

        # ---- Non-centered individual params ----
        with numpyro.plate("agents", n_agents):
            off_w = numpyro.sample("off_w", dist.Normal(0.0, 1.0))
            off_a = numpyro.sample("off_a", dist.Normal(0.0, 1.0))
            off_b = numpyro.sample("off_b", dist.Normal(0.0, 1.0))
            off_p = numpyro.sample("off_p", dist.Normal(0.0, 1.0))

        w_mb      = jax.nn.sigmoid(mu_w + sig_w * off_w)
        alpha_r   = jax.nn.sigmoid(mu_a + sig_a * off_a)
        beta_ch   = jnp.exp(mu_b + sig_b * off_b)
        phi_per   = mu_p + sig_p * off_p

        # Observations arrive as (T, N) — the scan output of simulate_observations.
        # scan slices the leading axis, so each step receives (N,) tensors.
        # Assert shapes to catch silent mismatches.
        assert a1_obs.shape == (n_ticks, n_agents), f"a1 shape {a1_obs.shape}"
        a1_T = a1_obs
        s2_T = s2_obs
        a2_T = a2_obs
        r_T  = r_obs

        carry0 = (Q_mf0, Q_s2_0, P_hat0, last0)

        def step(carry, xs):
            Q_mf, Q_s2, P_hat, last_a1 = carry
            a1_t, s2_t, a2_t, r_t = xs

            # w_mb_eff (distress is fixed baseline — not sampled here)
            w_mb_eff = jnp.clip(w_mb * (1.0 + C_b * E.C_BOOST)
                                * (1.0 - baseline * E.KAPPA_DECAY), 0.0, 1.0)

            # MB stage-1 values
            best_s2 = jnp.max(Q_s2, axis=-1)
            Q_MB = jnp.sum(P_hat * best_s2[:, None, :], axis=-1)
            Q_net = w_mb_eff[:, None] * Q_MB + (1.0 - w_mb_eff[:, None]) * Q_mf

            # Stage-1 logits
            engage_adv = E_bias - a_inhib
            avoid_adv  = theta_T + u_U
            persev = phi_per * jnp.where(last_a1 == 0, 1.0, 0.0)
            L1 = beta_ch[:, None] * jnp.stack(
                [Q_net[:, 0] + engage_adv + persev, Q_net[:, 1] + avoid_adv], axis=-1)
            L1 = jnp.clip(L1, -E.LOGIT_CLIP, E.LOGIT_CLIP)

            # Stage-1 likelihood
            logp1 = jax.nn.log_softmax(L1, axis=-1)
            ll_a1 = logp1[arange_N, a1_t]

            # Stage-2 logits — using OBSERVED s2
            q_s2_t = Q_s2[arange_N, s2_t, :]
            q_s2_t = jnp.where(q_s2_t < 0.0, q_s2_t * (1.0 - SP_b[:, None] * 0.5), q_s2_t)
            dark_bias = -SM_b - SN_b
            action_bonus = jnp.stack([dark_bias, dark_bias*0.5,
                                      jnp.zeros_like(dark_bias), jnp.zeros_like(dark_bias)], axis=-1)
            L2 = beta_ch[:, None] * (q_s2_t + action_bonus)
            L2 = jnp.clip(L2, -E.LOGIT_CLIP, E.LOGIT_CLIP)

            logp2 = jax.nn.log_softmax(L2, axis=-1)
            ll_a2 = logp2[arange_N, a2_t]

            # Q updates using OBSERVED actions
            q_s2_prev = Q_s2[arange_N, s2_t, a2_t]
            q_s2_new_val = jnp.clip(q_s2_prev + alpha_r*(r_t - q_s2_prev), -E.GRAD_CLIP, E.GRAD_CLIP)
            Q_s2_new = Q_s2.at[arange_N, s2_t, a2_t].set(q_s2_new_val)

            q_mf_prev = Q_mf[arange_N, a1_t]
            v_next = jnp.max(Q_mf, axis=-1)
            q_mf_new_val = jnp.clip(q_mf_prev + alpha_r*(r_t + gamma*v_next - q_mf_prev),
                                    -E.GRAD_CLIP, E.GRAD_CLIP)
            Q_mf_new = Q_mf.at[arange_N, a1_t].set(q_mf_new_val)

            # P_hat update (eta_vol is fixed)
            one_hot = jnp.stack([1.0 - s2_t.astype(jnp.float32), s2_t.astype(jnp.float32)], axis=-1)
            P_at = P_hat[arange_N, a1_t, :]
            P_upd = P_at + eta_vol[:, None] * (one_hot - P_at)
            P_upd = jnp.clip(P_upd, 1e-4, 1.0); P_upd = P_upd / jnp.sum(P_upd, axis=-1, keepdims=True)
            P_hat_new = P_hat.at[arange_N, a1_t, :].set(P_upd)

            last_a1_new = a1_t.astype(jnp.int32)
            return (Q_mf_new, Q_s2_new, P_hat_new, last_a1_new), (ll_a1, ll_a2)

        _, (ll_a1_seq, ll_a2_seq) = lax.scan(step, carry0, (a1_T, s2_T, a2_T, r_T))
        total_ll = jnp.sum(ll_a1_seq) + jnp.sum(ll_a2_seq)
        numpyro.factor("log_likelihood", total_ll)

        # Deterministic exposed quantities for post-processing
        numpyro.deterministic("w_mb_individual",      w_mb)
        numpyro.deterministic("alpha_r_individual",   alpha_r)
        numpyro.deterministic("beta_choice_individual", beta_ch)
        numpyro.deterministic("phi_per_individual",   phi_per)

    return model

# -----------------------------------------------------------------------------
# 3. Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("HBA FITTER — Full forward-simulated likelihood, NumPyro NUTS")
    print("=" * 70)
    print(f"Device: {jax.devices()[0]}")
    print(f"Config: N_AGENTS={N_AGENTS}  N_TICKS={N_TICKS}  "
          f"chains={N_CHAINS}  samples={N_SAMPLES}  warmup={N_WARMUP}")
    print()

    print("[1/3] Simulating observation dataset...")
    t0 = time.time()
    data = simulate_observations(N_AGENTS, N_TICKS, SEED)
    print(f"      wall = {time.time()-t0:.2f}s")
    for p in RECOVER_PARAMS:
        arr = data["true"][p]
        print(f"      true {p:14s}  mean={arr.mean():+.4f}  std={arr.std():.4f}  "
              f"min={arr.min():+.4f}  max={arr.max():+.4f}")

    print()
    print(f"[2/3] Building NumPyro model...")
    model = build_model(data["P"], N_AGENTS, N_TICKS)

    print(f"[3/3] Running NUTS MCMC (this will take several minutes)...")
    kernel = NUTS(model, target_accept_prob=0.9, max_tree_depth=8)
    mcmc = MCMC(kernel,
                num_warmup=N_WARMUP, num_samples=N_SAMPLES,
                num_chains=N_CHAINS, chain_method="sequential",
                progress_bar=True)
    t0 = time.time()
    mcmc.run(random.PRNGKey(SEED),
             a1_obs=jnp.asarray(data["a1"]),
             s2_obs=jnp.asarray(data["s2"]),
             a2_obs=jnp.asarray(data["a2"]),
             r_obs=jnp.asarray(data["r"]))
    print(f"      MCMC wall = {time.time()-t0:.2f}s")

    # ---- Recovery metrics ----
    samples = mcmc.get_samples()
    print()
    print("=" * 70)
    print("PARAMETER RECOVERY (posterior mean vs. true)")
    print("=" * 70)
    print(f"{'param':14s} {'rho':>8} {'p':>10} {'bias':>10} {'rmse':>10} {'verdict':>10}")
    print("-" * 70)
    results = {}
    for p in RECOVER_PARAMS:
        true_vals = data["true"][p]
        rec_vals = np.asarray(samples[f"{p if p!='w_MB' else 'w_mb'}_individual"].mean(axis=0)) \
                   if f"{p}_individual" in samples \
                   else np.asarray(samples[f"{p.lower()}_individual"].mean(axis=0))
        # resolve key aliases
        key_map = {"w_MB": "w_mb_individual", "alpha_R": "alpha_r_individual",
                   "beta_choice": "beta_choice_individual", "phi_per": "phi_per_individual"}
        rec_vals = np.asarray(samples[key_map[p]].mean(axis=0))
        rho, pval = spearmanr(true_vals, rec_vals)
        bias = float(np.mean(rec_vals - true_vals))
        rmse = float(np.sqrt(np.mean((rec_vals - true_vals)**2)))
        verdict = "PASS" if rho > 0.6 else "FAIL"
        results[p] = {"rho": float(rho), "p": float(pval),
                      "bias": bias, "rmse": rmse, "verdict": verdict}
        print(f"{p:14s} {rho:>8.4f} {pval:>10.2e} {bias:>10.4f} {rmse:>10.4f} {verdict:>10s}")

    n_pass = sum(1 for r in results.values() if r["verdict"] == "PASS")
    print()
    print("=" * 70)
    print(f"OVERALL: {n_pass}/{len(RECOVER_PARAMS)} parameters recovered with rho > 0.6")
    print("=" * 70)

    # Save
    with open(OUT / "hba_recovery_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT / 'hba_recovery_results.json'}")
