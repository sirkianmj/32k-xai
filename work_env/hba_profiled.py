# hba_profiled.py — Profiled identifiability test for engine_v5.
#
# For each of {w_MB, alpha_R, beta_choice, phi_per}, hold the other three
# at their true per-agent values and free ONLY the target. Fit a
# hierarchical model with 1 free parameter. If the posterior concentrates
# near the truth → conditionally identifiable. If not → no signal.
import os, time, json
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.stats import spearmanr
from jax import random, lax

import engine_v5 as E

N_AGENTS   = 60
N_TICKS    = 150
N_WARMUP   = 150
N_SAMPLES  = 300
N_CHAINS   = 1
SEED       = 42
OUT = Path(__file__).parent

# -----------------------------------------------------------------------------
# Simulate observation dataset (identical to hba_fitter)
# -----------------------------------------------------------------------------
def simulate(n_agents, n_ticks, seed):
    src = E.load_all(Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl")
    idx = np.random.default_rng(seed).integers(0, len(src["O"]), size=n_agents)
    P = {k: jnp.asarray(v[idx]) for k, v in src.items()}
    nb, _ = E.build_ws(n=n_agents, k=6, p=0.1, seed=0)
    neighbors = jnp.asarray(nb, jnp.int32)
    arange = jnp.arange(n_agents)
    valid_pick = jnp.where(neighbors >= 0, 0.0, -1e30)
    C_b     = P["C"]; eta_vol = P["eta_vol"]
    E_bias  = P["E"]*E.E_ENGAGE; a_inhib = P["a_inhib"]*E.A_INHIB
    theta_T = P["theta_T"]*E.T_THREAT; u_U = P["u_U"]*E.U_UNCERT
    SM_b    = P["S_M"]*E.SM_DEFECT; SP_b = P["S_P"]*E.SP_PUNISH; SN_b = P["S_N"]*E.SN_STATUS
    omega   = P["omega_social"]; lam_norm = P["lambda_norm"]*E.LAMBDA_NORM_A
    sal     = P["gamma_sal"]*E.SAL_GAIN; mood = P["m_t"]
    alpha_e = P["D"] + P["lambda_ineq"]; beta_e = P["A"]
    gamma   = 1.0/(1.0+jnp.exp(jnp.clip(P["k_DD"], -10.0, 5.0)))
    baseline= (P["general_distress"] + 0.3*P["DASS_Dep"] + 0.3*P["DASS_Anx"] + 0.3*P["DASS_Str"])/1.9
    lr = P["alpha_R"]; beta_s = P["beta_choice"]; phi = P["phi_per"]
    w_base = P["w_MB"]; N_s = P["N"]
    Q_mf0  = jnp.zeros((n_agents,2),jnp.float32)
    Q_s2_0 = jnp.zeros((n_agents,2,E.N_ACTIONS),jnp.float32)
    P_hat0 = jnp.zeros((n_agents,2,2),jnp.float32)
    P_hat0 = P_hat0.at[:,0,0].set(E.P_COMMON).at[:,0,1].set(1-E.P_COMMON)
    P_hat0 = P_hat0.at[:,1,0].set(1-E.P_COMMON).at[:,1,1].set(E.P_COMMON)
    w0 = jnp.full((n_agents,), E.WEALTH_INIT, jnp.float32)
    last0 = jnp.full((n_agents,), -1, jnp.int32)
    keys = random.split(random.PRNGKey(seed), n_ticks)
    def step(carry, tk):
        Q_mf, Q_s2, P_hat, wealth, dist, last_a1 = carry
        kp, kc, ka1, ka2, kr, _ = random.split(tk, 6)
        k_idx = random.categorical(kp, valid_pick, axis=-1)
        ch = neighbors[arange, k_idx]
        mut = (ch[ch] == arange)
        pid = jnp.minimum(arange, ch)
        ca  = random.bernoulli(kc, 0.5, shape=(n_agents,)).astype(jnp.float32)
        cp  = ca[pid]; il = arange < ch
        pp  = jnp.where(cp==1.0, il, ~il)
        isP = pp & mut; isR = (~pp) & mut
        w_eff = jnp.clip(w_base*(1.0+C_b*E.C_BOOST)*(1.0-dist*E.KAPPA_DECAY),0.0,1.0)
        best = jnp.max(Q_s2, axis=-1)
        Q_MB = jnp.sum(P_hat*best[:,None,:], axis=-1)
        Q_net = w_eff[:,None]*Q_MB + (1.0-w_eff[:,None])*Q_mf
        eng = E_bias - a_inhib; avd = theta_T + u_U
        per = phi*jnp.where(last_a1==0, 1.0, 0.0)
        L1 = beta_s[:,None]*jnp.stack([Q_net[:,0]+eng+per, Q_net[:,1]+avd], axis=-1)
        L1 = jnp.clip(L1, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a1 = random.categorical(ka1, L1, axis=-1)
        cm = random.bernoulli(kr, E.P_COMMON, shape=(n_agents,))
        s2 = jnp.where(cm, a1, 1-a1)
        qs = Q_s2[arange, s2, :]
        qs = jnp.where(qs<0.0, qs*(1.0-SP_b[:,None]*0.5), qs)
        db = -SM_b - SN_b
        ab = jnp.stack([db, db*0.5, jnp.zeros_like(db), jnp.zeros_like(db)], axis=-1)
        L2 = beta_s[:,None]*(qs + ab)
        L2 = jnp.clip(L2, -E.LOGIT_CLIP, E.LOGIT_CLIP)
        a2 = random.categorical(ka2, L2, axis=-1)
        my_off = E.ULTI_OFFERS[a2]; rec_off = my_off[ch]
        d_me = 2.0*rec_off - E.E_ENDOWMENT
        rp_me = jnp.maximum(d_me, 0.0); rn_me = jnp.maximum(-d_me, 0.0)
        my_UR = rec_off - alpha_e*rn_me - beta_e*rp_me + lam_norm
        my_acc = ((my_UR>=0.0) & isR).astype(jnp.float32)
        d_p = 2.0*my_off - E.E_ENDOWMENT
        rp_p = jnp.maximum(d_p, 0.0); rn_p = jnp.maximum(-d_p, 0.0)
        p_UR = my_off - alpha_e[ch]*rn_p - beta_e[ch]*rp_p
        p_acc = ((p_UR>=0.0) & isP).astype(jnp.float32)
        r_ultiP = jnp.where(p_acc>=1.0, E.E_ENDOWMENT-my_off, 0.0)
        r_ultiR = jnp.where(my_acc>=1.0, rec_off, 0.0)
        my_s = E.TRUST_SENDS[a2]; rec_s = my_s[ch]
        my_rm = E.TRUST_RETURN_MULT[a2]; rec_rm = my_rm[ch]
        r_ts = jnp.where(isR, -my_s + E.TRUST_K*rec_rm*my_s, 0.0)
        r_tt = jnp.where(isP, E.TRUST_K*rec_s*(1.0-my_rm/E.TRUST_K), 0.0)
        raw = r_ultiP + r_ultiR + r_ts + r_tt
        rew = jnp.clip(raw*omega*(1.0+sal+0.05*mood), -E.REWARD_CLIP, E.REWARD_CLIP)
        inter = (a1==0) & (a1[ch]==0) & mut
        rew = jnp.where(inter, rew, 0.0)
        um = inter.astype(jnp.float32)
        qsv = Q_s2[arange,s2,a2]
        Q_s2n = Q_s2.at[arange,s2,a2].set(jnp.clip(qsv+um*lr*(rew-qsv), -E.GRAD_CLIP, E.GRAD_CLIP))
        qmv = Q_mf[arange,a1]; vn = jnp.max(Q_mf, axis=-1)
        Q_mfn = Q_mf.at[arange,a1].set(jnp.clip(qmv+um*lr*(rew+gamma*vn-qmv), -E.GRAD_CLIP, E.GRAD_CLIP))
        oh = jnp.stack([1.0-s2.astype(jnp.float32), s2.astype(jnp.float32)], axis=-1)
        Pat = P_hat[arange,a1,:]
        Pu = Pat + um[:,None]*eta_vol[:,None]*(oh-Pat)
        Pu = jnp.clip(Pu, 1e-4, 1.0); Pu = Pu/jnp.sum(Pu, axis=-1, keepdims=True)
        P_hatn = P_hat.at[arange,a1,:].set(Pu)
        rej = isR & inter & (my_UR<0.0)
        dn = E.DISTRESS_DECAY*dist + E.DISTRESS_PULL*baseline + (1.0+N_s*E.N_DISTRESS)*rej.astype(jnp.float32)*0.05
        wn = jnp.maximum(wealth+rew, 0.0)
        return (Q_mfn, Q_s2n, P_hatn, wn, dn, a1), (a1, s2, a2, rew, um, dn)
    @jax.jit
    def run(c, ks): return lax.scan(step, c, ks)
    _, (a1, s2, a2, r, um, dist_tr) = run((Q_mf0,Q_s2_0,P_hat0,w0,baseline,last0), keys)
    jax.block_until_ready(a1)
    return {"a1": np.asarray(a1, np.int32), "s2": np.asarray(s2, np.int32),
            "a2": np.asarray(a2, np.int32), "r": np.asarray(r, np.float32),
            "interacting": np.asarray(um, np.float32),
            "distress":    np.asarray(dist_tr, np.float32),
            "P": P}

# -----------------------------------------------------------------------------
# Profiled model — only ONE parameter free
# -----------------------------------------------------------------------------
def build_profiled_model(P_fixed, target, n_agents, n_ticks):
    # Extract fixed per-agent quantities
    C_b     = P_fixed["C"]; eta_vol = P_fixed["eta_vol"]
    E_bias  = P_fixed["E"]*E.E_ENGAGE; a_inhib = P_fixed["a_inhib"]*E.A_INHIB
    theta_T = P_fixed["theta_T"]*E.T_THREAT; u_U = P_fixed["u_U"]*E.U_UNCERT
    SM_b    = P_fixed["S_M"]*E.SM_DEFECT; SP_b = P_fixed["S_P"]*E.SP_PUNISH; SN_b = P_fixed["S_N"]*E.SN_STATUS
    gamma   = 1.0/(1.0+jnp.exp(jnp.clip(P_fixed["k_DD"], -10.0, 5.0)))
    baseline= (P_fixed["general_distress"] + 0.3*P_fixed["DASS_Dep"]
               + 0.3*P_fixed["DASS_Anx"] + 0.3*P_fixed["DASS_Str"])/1.9
    # True values of the three non-targets (hold at truth)
    fix_w = P_fixed["w_MB"]; fix_a = P_fixed["alpha_R"]
    fix_b = P_fixed["beta_choice"]; fix_p = P_fixed["phi_per"]
    arange = jnp.arange(n_agents)
    Q_mf0  = jnp.zeros((n_agents,2),jnp.float32)
    Q_s2_0 = jnp.zeros((n_agents,2,E.N_ACTIONS),jnp.float32)
    P_hat0 = jnp.zeros((n_agents,2,2),jnp.float32)
    P_hat0 = P_hat0.at[:,0,0].set(E.P_COMMON).at[:,0,1].set(1-E.P_COMMON)
    P_hat0 = P_hat0.at[:,1,0].set(1-E.P_COMMON).at[:,1,1].set(E.P_COMMON)
    last0  = jnp.full((n_agents,), -1, jnp.int32)

    def model(a1_obs, s2_obs, a2_obs, r_obs, interact_obs, distress_obs):
        # One hyper-prior + one non-centered offset per agent
        mu_t  = numpyro.sample("mu_t",  dist.Normal(0.0, 1.0))
        sig_t = numpyro.sample("sig_t", dist.HalfNormal(1.0))
        with numpyro.plate("agents", n_agents):
            off_t = numpyro.sample("off_t", dist.Normal(0.0, 1.0))
        raw = mu_t + sig_t * off_t
        if target == "w_MB":
            w_mb = jax.nn.sigmoid(raw); alpha_r = fix_a; beta_ch = fix_b; phi_per = fix_p
        elif target == "alpha_R":
            alpha_r = jax.nn.sigmoid(raw); w_mb = fix_w; beta_ch = fix_b; phi_per = fix_p
        elif target == "beta_choice":
            beta_ch = jnp.exp(raw); w_mb = fix_w; alpha_r = fix_a; phi_per = fix_p
        else:  # phi_per
            phi_per = raw; w_mb = fix_w; alpha_r = fix_a; beta_ch = fix_b

        a1_T = a1_obs; s2_T = s2_obs; a2_T = a2_obs; r_T = r_obs
        um_T = interact_obs; dist_T = distress_obs
        carry0 = (Q_mf0, Q_s2_0, P_hat0, last0)

        def step(carry, xs):
            Q_mf, Q_s2, P_hat, last_a1 = carry
            a1_t, s2_t, a2_t, r_t, um_t, dist_t = xs
            w_eff = jnp.clip(w_mb*(1.0+C_b*E.C_BOOST)*(1.0-dist_t*E.KAPPA_DECAY), 0.0, 1.0)
            best = jnp.max(Q_s2, axis=-1)
            Q_MB = jnp.sum(P_hat*best[:,None,:], axis=-1)
            Q_net = w_eff[:,None]*Q_MB + (1.0-w_eff[:,None])*Q_mf
            eng = E_bias - a_inhib; avd = theta_T + u_U
            per = phi_per*jnp.where(last_a1==0, 1.0, 0.0)
            L1 = beta_ch[:,None]*jnp.stack([Q_net[:,0]+eng+per, Q_net[:,1]+avd], axis=-1)
            L1 = jnp.clip(L1, -E.LOGIT_CLIP, E.LOGIT_CLIP)
            logp1 = jax.nn.log_softmax(L1, axis=-1)
            ll1 = logp1[arange, a1_t]
            qs_t = Q_s2[arange, s2_t, :]
            qs_t = jnp.where(qs_t<0.0, qs_t*(1.0-SP_b[:,None]*0.5), qs_t)
            db = -SM_b - SN_b
            ab = jnp.stack([db, db*0.5, jnp.zeros_like(db), jnp.zeros_like(db)], axis=-1)
            L2 = beta_ch[:,None]*(qs_t + ab)
            L2 = jnp.clip(L2, -E.LOGIT_CLIP, E.LOGIT_CLIP)
            logp2 = jax.nn.log_softmax(L2, axis=-1)
            ll2 = logp2[arange, a2_t]
            qsv = Q_s2[arange,s2_t,a2_t]
            Q_s2n = Q_s2.at[arange,s2_t,a2_t].set(jnp.clip(qsv+um_t*alpha_r*(r_t-qsv), -E.GRAD_CLIP, E.GRAD_CLIP))
            qmv = Q_mf[arange,a1_t]; vn = jnp.max(Q_mf, axis=-1)
            Q_mfn = Q_mf.at[arange,a1_t].set(jnp.clip(qmv+um_t*alpha_r*(r_t+gamma*vn-qmv), -E.GRAD_CLIP, E.GRAD_CLIP))
            oh = jnp.stack([1.0-s2_t.astype(jnp.float32), s2_t.astype(jnp.float32)], axis=-1)
            Pat = P_hat[arange,a1_t,:]
            Pu = Pat + um_t[:,None]*eta_vol[:,None]*(oh-Pat)
            Pu = jnp.clip(Pu, 1e-4, 1.0); Pu = Pu/jnp.sum(Pu, axis=-1, keepdims=True)
            P_hatn = P_hat.at[arange,a1_t,:].set(Pu)
            return (Q_mfn, Q_s2n, P_hatn, a1_t.astype(jnp.int32)), (ll1, ll2)

        _, (ll1_s, ll2_s) = lax.scan(step, carry0, (a1_T, s2_T, a2_T, r_T, um_T, dist_T))
        numpyro.factor("ll", jnp.sum(ll1_s) + jnp.sum(ll2_s))
        if target == "w_MB":
            numpyro.deterministic("target_ind", w_mb)
        elif target == "alpha_R":
            numpyro.deterministic("target_ind", alpha_r)
        elif target == "beta_choice":
            numpyro.deterministic("target_ind", beta_ch)
        else:
            numpyro.deterministic("target_ind", phi_per)
    return model

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("PROFILED IDENTIFIABILITY TEST")
    print("="*70)
    print(f"Device: {jax.devices()[0]}  N={N_AGENTS} T={N_TICKS}")
    print()
    print("[sim] generating observation dataset...")
    t0 = time.time()
    data = simulate(N_AGENTS, N_TICKS, SEED)
    print(f"      wall = {time.time()-t0:.2f}s")

    truth = {p: np.asarray(data["P"][p]) for p in
             ["w_MB","alpha_R","beta_choice","phi_per"]}

    results = {}
    for target in ["w_MB", "alpha_R", "beta_choice", "phi_per"]:
        print()
        print(f"--- target: {target} ---")
        model = build_profiled_model(data["P"], target, N_AGENTS, N_TICKS)
        kernel = NUTS(model, target_accept_prob=0.85, max_tree_depth=6)
        mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES,
                    num_chains=N_CHAINS, progress_bar=True)
        t0 = time.time()
        mcmc.run(random.PRNGKey(SEED),
                 a1_obs=jnp.asarray(data["a1"]),
                 s2_obs=jnp.asarray(data["s2"]),
                 a2_obs=jnp.asarray(data["a2"]),
                 r_obs=jnp.asarray(data["r"]),
                 interact_obs=jnp.asarray(data["interacting"]),
                 distress_obs=jnp.asarray(data["distress"]))
        dt = time.time()-t0
        samples = mcmc.get_samples()
        rec = np.asarray(samples["target_ind"].mean(axis=0))
        true = truth[target]
        rho, pval = spearmanr(true, rec)
        bias = float(np.mean(rec - true))
        rmse = float(np.sqrt(np.mean((rec - true)**2)))
        verdict = "IDENTIFIABLE" if rho > 0.6 else "NOT-IDENTIFIABLE"
        results[target] = {"rho": float(rho), "p": float(pval),
                           "bias": bias, "rmse": rmse, "wall": dt,
                           "verdict": verdict}
        print(f"      wall={dt:.1f}s  rho={rho:.4f}  bias={bias:+.4f}  rmse={rmse:.4f}  {verdict}")

    print()
    print("="*70)
    print("PROFILED IDENTIFIABILITY SUMMARY")
    print("="*70)
    print(f"{'param':14s} {'rho':>8} {'bias':>10} {'rmse':>10} {'verdict':>18}")
    print("-"*70)
    for t, r in results.items():
        print(f"{t:14s} {r['rho']:>8.4f} {r['bias']:>+10.4f} {r['rmse']:>10.4f} {r['verdict']:>18s}")
    print("="*70)
    n_id = sum(1 for r in results.values() if r["verdict"] == "IDENTIFIABLE")
    print(f"{n_id}/4 parameters identifiable when others are held at truth.")

    with open(OUT / "hba_profiled_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT / 'hba_profiled_results.json'}")
