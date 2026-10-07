"""
G2 — Evidence-Constrained Hierarchical Copula (per-coupling literature SE)

Architecture: 28 atomic + 2 deterministic composites.
Per-coupling literature SE read from CSV column `se_lit`.
"""
import numpy as np, pandas as pd, pickle, time, warnings
from pathlib import Path
from scipy.stats import norm, kstest, wasserstein_distance
warnings.filterwarnings("ignore")

ROOT = Path.home() / "projects/xai_project"
OUT  = ROOT / "work_g2"
N_AGENTS = 32000
SEED = 20261007

CONFIDENCE_WEIGHT = {
    "empirical_direct":          1.0,
    "literature_meta_analysis":  0.9,
    "derived":                   0.7,
    "empirical_null":            0.7,
    "direction_supported":       0.4,
    "indirect_evidence":         0.3,
    "unverified":                0.0,
}
DISAGG_DISCOUNT = 0.8

# -------------------------------------------------------------------------
# Disaggregation rules
# -------------------------------------------------------------------------
def rule_A_equal_cov(rho_XY, sigma_Y, sigmas_C):
    return [rho_XY * sigma_Y / s for s in sigmas_C]

def rule_B_uniform_corr(rho_XY, sigma_Y, sigmas_C):
    k = len(sigmas_C)
    c = k * rho_XY * sigma_Y / sum(sigmas_C)
    return [c] * k

def rule_C_variance_prop(rho_XY, sigma_Y, sigmas_C):
    k = len(sigmas_C)
    sum_sq = sum(s**2 for s in sigmas_C)
    k_C = k * rho_XY * sigma_Y / sum_sq
    return [k_C * s for s in sigmas_C]

RULES = {
    "A_equal_cov":     rule_A_equal_cov,
    "B_uniform_corr":  rule_B_uniform_corr,
    "C_variance_prop": rule_C_variance_prop,
}

# =========================================================================
# Load source marginals
# =========================================================================
print("=== Loading source marginals ===", flush=True)
S = {}
ocean = pd.read_csv(ROOT / "work_ipip/ipip_ocean_ipc1.csv")
for c in ["O","C","E","A","N"]: S[c] = ocean[c].to_numpy()
sd3 = pd.read_csv(ROOT / "work_sd3/sd3_subscales.csv")
S["S_M"] = sd3["Mach_n"].to_numpy()
S["S_P"] = sd3["Psyc_n"].to_numpy()
S["S_N"] = sd3["Narc_n"].to_numpy()
dass = pd.read_csv(ROOT / "work_dass/dass42_subscales.csv")
S["DASS_Dep"] = dass["D42_Dep_n"].to_numpy()
S["DASS_Anx"] = dass["D42_Anx_n"].to_numpy()
S["DASS_Str"] = dass["D42_Str_n"].to_numpy()
ts = pd.read_csv(ROOT / "work_n500/w_mb_real.csv")
S["alpha_R"]     = ts["alpha"].to_numpy()
S["beta_choice"] = ts["beta"].to_numpy()
S["phi_per"]     = ts["persev"].to_numpy()
S["w_MB"]        = ts["w"].to_numpy()
dd = pd.read_csv(ROOT / "work_dd/dd_k_subjects.csv")
S["k_DD"] = dd["log_k"].to_numpy()
ss = pd.read_csv(ROOT / "work_ssrt/ssrt_subjects.csv")
S["a_inhib"] = ss["ssrt"].dropna().to_numpy()
fs = pd.read_csv(ROOT / "work_fs/fs_alpha_subjects.csv")
S["lambda_ineq"] = fs["alpha"].to_numpy()

rng_syn = np.random.default_rng(SEED); N_SYNTH = 20000
S["theta_T"]      = np.clip(rng_syn.lognormal(0.0,  0.5, N_SYNTH), 0.01, 3.0)
S["u_U"]          = np.clip(rng_syn.lognormal(0.0,  0.5, N_SYNTH), 0.01, 3.0)
S["eta_vol"]      = np.clip(rng_syn.lognormal(-0.7, 0.5, N_SYNTH), 0.01, 1.0)
S["omega_social"] = np.clip(rng_syn.lognormal(0.0,  0.5, N_SYNTH), 0.01, 3.0)
S["lambda_norm"]  = np.clip(rng_syn.lognormal(0.0,  0.5, N_SYNTH), 0.01, 3.0)
S["s_spite"]      = np.clip(rng_syn.lognormal(-0.7, 0.5, N_SYNTH), 0.01, 3.0)
S["tau_sens"]     = np.clip(rng_syn.lognormal(0.0,  0.5, N_SYNTH), 0.01, 3.0)
S["gamma_sal"]    = np.clip(rng_syn.lognormal(0.0,  0.5, N_SYNTH), 0.01, 3.0)
S["kappa_ctx"]    = rng_syn.beta(2, 2, N_SYNTH)
S["m_t"]          = rng_syn.normal(0.0, 1.0, N_SYNTH)

composite_components = {
    "D":                ["S_M", "S_P", "S_N"],
    "general_distress": ["DASS_Dep", "DASS_Anx", "DASS_Str"],
}
src_std = {k: float(np.std(v, ddof=1)) for k, v in S.items()}
for comp, parts in composite_components.items():
    stacked = np.column_stack([S[p] for p in parts])
    src_std[comp] = float(np.std(stacked.mean(axis=1), ddof=1))

ATOMIC = [k for k in S.keys() if k not in composite_components]
d = len(ATOMIC); idx = {n: i for i, n in enumerate(ATOMIC)}
print(f"  Atomic dim = {d}, composites = {list(composite_components.keys())}")

couplings = pd.read_csv(OUT / "literature_couplings/pairwise_correlations_verified.csv")

def project_to_corr(M):
    w_eig, V = np.linalg.eigh((M + M.T) * 0.5)
    w_eig = np.clip(w_eig, 1e-8, None)
    Mn = V @ np.diag(w_eig) @ V.T
    dg = np.sqrt(np.diag(Mn))
    Mn = Mn / np.outer(dg, dg)
    np.fill_diagonal(Mn, 1.0)
    return Mn

def run_pipeline(rule_name, allocate_fn):
    print(f"\n  RUN: {rule_name}", flush=True)
    R = np.eye(d); W = np.zeros((d, d))
    FLAGS = np.full((d, d), "NO_EVIDENCE", dtype=object)
    for i in range(d): FLAGS[i, i] = "SELF"
    for _, row in couplings.iterrows():
        a, b = row["variable_1"], row["variable_2"]
        status = str(row["status"]).strip().lower()
        if status == "derived": continue
        base_w = CONFIDENCE_WEIGHT.get(status, 0.0)
        if base_w == 0.0: continue
        rho = row["rho"]
        if a in idx and b in idx and pd.notna(rho):
            i, j = idx[a], idx[b]
            R[i, j] = R[j, i] = float(rho)
            W[i, j] = W[j, i] = base_w
            FLAGS[i, j] = FLAGS[j, i] = status.upper()
            continue
        if (a in composite_components or b in composite_components) and pd.notna(rho):
            comp_name = a if a in composite_components else b
            other = b if comp_name == a else a
            if other not in idx: continue
            parts = composite_components[comp_name]
            sigmas_C = [src_std[p] for p in parts]
            allocated = allocate_fn(float(rho), src_std[comp_name], sigmas_C)
            for p, rho_di in zip(parts, allocated):
                if p not in idx: continue
                i, j = idx[other], idx[p]
                if FLAGS[i, j] == "NO_EVIDENCE":
                    R[i, j] = R[j, i] = rho_di
                    W[i, j] = W[j, i] = base_w * DISAGG_DISCOUNT
                    FLAGS[i, j] = FLAGS[j, i] = "DISAGG_" + status.upper()
                else:
                    w_old = W[i, j]; w_new = base_w * DISAGG_DISCOUNT
                    R[i, j] = R[j, i] = (R[i, j]*w_old + rho_di*w_new) / (w_old + w_new)
                    W[i, j] = W[j, i] = max(W[i, j], w_new)
            continue
        if pd.isna(rho):
            if status == "empirical_null" and a in idx and b in idx:
                i, j = idx[a], idx[b]
                R[i, j] = R[j, i] = 0.0
                W[i, j] = W[j, i] = base_w
                FLAGS[i, j] = FLAGS[j, i] = "EMPIRICAL_NULL"
            continue

    np.fill_diagonal(R, 1.0); np.fill_diagonal(W, 0.0)
    R_target = R.copy(); R_work = R.copy()
    for _ in range(60):
        R_work = project_to_corr(R_work)
        R_work = R_work + W * (R_target - R_work)
        np.fill_diagonal(R_work, 1.0)
    R_work = project_to_corr(R_work)
    R_Spearman = R_work
    R_Pearson = 2.0 * np.sin(np.pi * R_Spearman / 6.0)
    np.fill_diagonal(R_Pearson, 1.0)
    R_Pearson = project_to_corr(R_Pearson)

    rng = np.random.default_rng(SEED + 1)
    L = np.linalg.cholesky(R_Pearson + 1e-10 * np.eye(d))
    Z = rng.standard_normal((N_AGENTS, d)) @ L.T
    U = norm.cdf(Z)
    X = np.zeros((N_AGENTS, d))
    for name in ATOMIC:
        x_src = S[name]; n_src = len(x_src)
        u_col = U[:, idx[name]]; order = np.argsort(u_col)
        if n_src >= N_AGENTS:
            sub = rng.choice(n_src, size=N_AGENTS, replace=False)
        else:
            sub = rng.choice(n_src, size=N_AGENTS, replace=True)
        X[order, idx[name]] = np.sort(x_src[sub])
    df = pd.DataFrame(X, columns=ATOMIC)
    for comp, parts in composite_components.items():
        df[comp] = df[parts].mean(axis=1)

    g21_ok = True
    for name in ATOMIC:
        x_src = S[name]; x_syn = df[name].to_numpy()
        ks_stat, _ = kstest(x_src, x_syn, alternative="two-sided")
        rel_w = wasserstein_distance(x_src, x_syn) / (np.std(x_src) + 1e-12)
        if not ((ks_stat < 0.04) and (rel_w < 0.08)): g21_ok = False

    g22_ok = True
    for comp, parts in composite_components.items():
        diff = float(np.max(np.abs(df[comp].to_numpy() - df[parts].mean(axis=1).to_numpy())))
        if diff >= 1e-10: g22_ok = False

    R_syn = df.corr(method="spearman").to_numpy()
    names_ext = list(df.columns); idx_ext = {n: i for i, n in enumerate(names_ext)}
    comp_coupling = {}
    for _, row in couplings.iterrows():
        a, b = row["variable_1"], row["variable_2"]
        if str(row["status"]).strip().lower() == "derived": continue
        if pd.isna(row["rho"]): continue
        if a not in idx_ext or b not in idx_ext: continue
        if not (a in composite_components or b in composite_components): continue
        comp_coupling[(a, b)] = float(R_syn[idx_ext[a], idx_ext[b]])

    return {
        "rule": rule_name, "g21_ok": g21_ok, "g22_ok": g22_ok,
        "composite_couplings": comp_coupling,
        "df": df, "R_Spearman": R_Spearman, "W": W, "FLAGS": FLAGS,
    }

results = {}
for rule_name, fn in RULES.items():
    results[rule_name] = run_pipeline(rule_name, fn)

# =========================================================================
# Sensitivity analysis with per-coupling literature SE
# =========================================================================
print("\n\n" + "#"*72)
print("# SENSITIVITY — composite couplings with per-coupling literature SE")
print("#"*72)

all_targets = set()
for r in results.values():
    all_targets |= set(r["composite_couplings"].keys())
all_targets = sorted(all_targets)

# Build target_of AND se_of dicts
target_of = {}; se_of = {}
for _, row in couplings.iterrows():
    a, b = row["variable_1"], row["variable_2"]
    for key in [(a,b), (b,a)]:
        if key in all_targets:
            target_of[key] = float(row["rho"]) if pd.notna(row["rho"]) else np.nan
            se_val = row.get("se_lit", np.nan)
            se_of[key] = float(se_val) if pd.notna(se_val) else 0.03

se_syn = 1.0 / np.sqrt(N_AGENTS)
rows = []
for tgt in all_targets:
    a, b = tgt
    tgt_val = target_of.get(tgt, np.nan)
    se_val = se_of.get(tgt, 0.03)
    rule_vals = [results[r]["composite_couplings"].get(tgt, np.nan) for r in RULES]
    spread = float(np.nanmax(rule_vals) - np.nanmin(rule_vals))
    tol_combined = 2.0 * np.sqrt(se_val**2 + se_syn**2)
    mean_val = float(np.nanmean(rule_vals))
    err_to_target = abs(mean_val - tgt_val)
    rows.append({
        "var_1": a, "var_2": b, "target": tgt_val, "se_lit": se_val,
        "A": rule_vals[0], "B": rule_vals[1], "C": rule_vals[2],
        "spread": spread, "tol_combined": tol_combined,
        "mean_err": err_to_target,
        "robust": spread < tol_combined,
        "compatible": err_to_target < tol_combined,
    })
sens = pd.DataFrame(rows)
print(sens.round(4).to_string(index=False))
sens.to_csv(OUT / "sensitivity_composite_couplings.csv", index=False)

g23_robust = bool(sens["robust"].all())
g23_compatible = bool(sens["compatible"].all())

print(f"\n" + "="*72)
print("  FINAL G2 VERDICT (per-coupling literature SE)")
print("="*72)
primary = results["A_equal_cov"]
g21_pass = primary["g21_ok"]
g22_pass = primary["g22_ok"]
print(f"  G2.1 atomic marginals:              {'PASS' if g21_pass else 'FAIL'}")
print(f"  G2.2 structural consistency:        {'PASS' if g22_pass else 'FAIL'}")
print(f"  G2.3a robustness across rules:      {'PASS' if g23_robust else 'FAIL'}")
print(f"  G2.3b literature compatibility:     {'PASS' if g23_compatible else 'FAIL'}")
overall = "PASS" if (g21_pass and g22_pass and g23_robust and g23_compatible) else "FAIL"
print(f"  OVERALL: {overall}")

with open(OUT / "validation.txt", "w") as f:
    f.write(f"N_AGENTS={N_AGENTS}\natomic_d={d}\n")
    f.write(f"g21_pass={g21_pass}\ng22_pass={g22_pass}\n")
    f.write(f"g23_robust={g23_robust}\ng23_compatible={g23_compatible}\n")
    f.write(f"overall={overall}\n")
    f.write(f"se_synth={se_syn:.5f}\n")
    f.write(f"per_coupling_lit_se=yes\n")

primary["df"].to_pickle(OUT / "synthetic_32k.pkl")
primary["df"].sample(n=min(50000, N_AGENTS), random_state=SEED).to_csv(
    OUT / "synthetic_32k_sample50k.csv", index=False)
np.save(OUT / "target_Spearman.npy", primary["R_Spearman"])
pickle.dump({"names": ATOMIC, "flags": primary["FLAGS"], "weights": primary["W"]},
            open(OUT / "target_R_flags.pkl", "wb"))
print(f"\nSaved canonical Rule A outputs to {OUT}")
