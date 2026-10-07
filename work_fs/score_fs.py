import pandas as pd, numpy as np, time, pickle, warnings
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
DATA = Path.home() / "projects/xai_project/Data_Xai32kProje/XAI_Social_Games_Data"
OUT  = Path.home() / "projects/xai_project/work_fs"

def sigmoid(x):
    # numerically stable
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

# ---- Parse ----------------------------------------------------------------
t0 = time.time()
rows = []
for f in sorted(DATA.glob("sub-*_task-ugr_*.tsv")):
    sid = f.name.split("_")[0]
    try: df = pd.read_csv(f, sep="\t")
    except Exception: continue
    dec = df[df["trial_type"].astype(str).str.startswith("dec")].copy()
    for onset in dec["onset"].unique():
        tr = dec[dec["onset"] == onset]
        tt = set(tr["trial_type"].astype(str))
        is_acc = any("accept" in t for t in tt)
        is_rej = any("reject" in t for t in tt)
        if is_acc == is_rej: continue
        rows.append({"subject": sid,
                     "endowment": float(tr["Endowment"].iloc[0]),
                     "offer": float(tr["Decision"].iloc[0]),
                     "accept": 1 if is_acc else 0})
trials = pd.DataFrame(rows)
print(f"Parsed: {len(trials)} trials, {trials['subject'].nunique()} subjects, {time.time()-t0:.1f}s", flush=True)

# ---- Model ----------------------------------------------------------------
def neg_ll(params, offer, endow, acc):
    alpha, tau = params
    U = offer - alpha * (endow - 2.0 * offer)
    p = sigmoid(tau * U)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(acc * np.log(p) + (1 - acc) * np.log(1 - p))

def fit_one(offer, endow, acc, seed=0, n_restart=3):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_restart):
        x0 = [rng.uniform(0.05, 2.0), rng.uniform(0.5, 5.0)]
        r = minimize(neg_ll, x0, args=(offer, endow, acc),
                     method="L-BFGS-B",
                     bounds=[(0.0, 10.0), (1e-3, 50.0)],
                     options={"maxiter": 100, "ftol": 1e-8})
        if best is None or r.fun < best.fun: best = r
    return best.x

# ---- Real fits ------------------------------------------------------------
subs = sorted(trials["subject"].unique())
res = []
for i, s in enumerate(subs):
    tr = trials[trials["subject"] == s]
    if len(tr) < 5: continue
    a, t = fit_one(tr["offer"].to_numpy(), tr["endowment"].to_numpy(),
                    tr["accept"].to_numpy(), seed=1000+i)
    res.append({"subject": s, "n_trials": len(tr), "alpha": a, "tau": t,
                "accept_rate": float(tr["accept"].mean()),
                "mean_offer": float(tr["offer"].mean())})
real = pd.DataFrame(res)
print(f"Fits done: {len(real)} subjects, {time.time()-t0:.1f}s", flush=True)
print()
print("=== Real-data estimates ===")
print(real[["alpha","tau","accept_rate","n_trials"]].describe().round(4).to_string())
print()
print(f"alpha at lower bound 0:  {(real['alpha'] < 0.01).sum()} / {len(real)}")
print(f"alpha at upper bound 10: {(real['alpha'] > 9.99).sum()} / {len(real)}")
print(f"corr(alpha, accept_rate) = {real['alpha'].corr(real['accept_rate']):.4f}")
print(flush=True)

# ---- Recovery -------------------------------------------------------------
print(f"\n=== Recovery test N=500 ===", flush=True)
rng = np.random.default_rng(42)
N_SIM = 500
true_alpha = rng.uniform(0.05, 3.0, N_SIM)
true_tau   = rng.uniform(0.5, 5.0, N_SIM)
OFF_16 = [0.0, 1.0, 2.0, 4.0, 8.0]
OFF_32 = [0.0, 2.0, 3.0, 8.0, 16.0]

def simulate(alpha, tau, n, seed):
    r_ = np.random.default_rng(seed)
    E = r_.choice([16.0, 32.0], size=n)
    offers = np.where(E == 16, r_.choice(OFF_16, size=n), r_.choice(OFF_32, size=n))
    U = offers - alpha * (E - 2.0 * offers)
    acc = (r_.random(n) < sigmoid(tau * U)).astype(int)
    return offers, E, acc

t1 = time.time()
rec = np.full((N_SIM, 2), np.nan)
for i in range(N_SIM):
    if i % 50 == 0: print(f"  sim {i}/{N_SIM}  elapsed {time.time()-t1:.1f}s", flush=True)
    n = int(rng.integers(40, 120))
    o, e, a = simulate(true_alpha[i], true_tau[i], n, seed=2000+i)
    if a.sum() == 0 or a.sum() == len(a): continue
    ah, th = fit_one(o, e, a, seed=3000+i, n_restart=2)
    rec[i] = [ah, th]

mask = ~np.isnan(rec[:, 0])
rho_a, p_a = spearmanr(true_alpha[mask], rec[mask, 0])
bias_a = float(np.mean(rec[mask, 0] - true_alpha[mask]))
rmse_a = float(np.sqrt(np.mean((rec[mask, 0] - true_alpha[mask]) ** 2)))

print(f"\nN_valid = {mask.sum()}/{N_SIM}")
print(f"alpha recovery: rho={rho_a:.4f}  p={p_a:.3e}  bias={bias_a:.4f}  rmse={rmse_a:.4f}")
print(f"Pass (rho>0.5): {'YES' if rho_a > 0.5 else 'NO'}")

real.to_csv(OUT / "fs_alpha_subjects.csv", index=False)
with open(OUT / "fs_alpha_subjects.pkl", "wb") as fh: pickle.dump(real, fh, protocol=4)
trials.to_csv(OUT / "ugr_trials.csv", index=False)
with open(OUT / "fs_recovery.txt", "w") as fh:
    fh.write(f"N_SIM={N_SIM}\nN_valid={int(mask.sum())}\n"
             f"rho_alpha={rho_a:.4f}\np_alpha={p_a:.3e}\n"
             f"bias_alpha={bias_a:.4f}\nrmse_alpha={rmse_a:.4f}\n")
print(f"\nSaved to {OUT}")
