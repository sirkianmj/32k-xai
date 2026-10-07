"""
G5 — Kool novel two-step paradigm (w_MB recovery) — parallelized
"""
import numpy as np, pandas as pd, pickle, time, warnings
from pathlib import Path
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import spearmanr
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

ROOT  = Path.home() / "projects/xai_project"
TRADE = ROOT / "tradeoffs/data/novel paradigm/data.mat"
OUT   = ROOT / "work_kool"
N_JOBS = -1

# ---------- Load ----------------------------------------------------------
print("=== Loading ===", flush=True)
m = loadmat(str(TRADE), simplify_cells=True)
raw = m["data"]
subjects = {}
for row in raw:
    subjects.setdefault(row[0], []).append(row[1:])

sub_data = {}
for sid, rows in subjects.items():
    arr = np.array(rows, dtype=object)
    practice = np.array([int(r[9]) for r in rows])
    keep = practice == 0
    if keep.sum() < 20: continue
    sub_data[sid] = arr[keep].astype(float)
print(f"  {len(sub_data)} subjects loaded", flush=True)

# ---------- Model ---------------------------------------------------------
def kool_negll(params, subdata):
    b, lr, lam, w = params
    if not (0.01 < b < 20): return 1e10
    if not (0.01 < lr < 0.99): return 1e10
    if not (0.01 < lam < 0.99): return 1e10
    if not (0.0 <= w <= 1.0): return 1e10

    Qmf = np.zeros((2, 2)); Q2 = np.zeros(2)
    Tm = [np.full((2,2), 0.5), np.full((2,2), 0.5)]
    Tmchanged = [False, False]
    R = np.zeros(2)
    LL = 0.0
    for t in range(len(subdata)):
        s1 = int(subdata[t, 0]); stim_l = int(subdata[t, 1])
        rt1 = subdata[t, 3]; c1 = int(subdata[t, 4]); rt2 = subdata[t, 5]
        points = subdata[t, 6]
        if rt1 == -1 or rt2 == -1: continue
        if stim_l == 2 or stim_l == 4: R = np.flipud(R).copy()
        a = (c1 - 1) % 2
        Qmb = Tm[s1 - 1].T @ Q2
        Q = w * Qmb + (1 - w) * Qmf[s1 - 1, :]
        LL += b * Q[a] - logsumexp(b * Q)
        if not Tmchanged[s1 - 1]:
            Tmchanged[s1 - 1] = True; Tm[s1 - 1] = np.eye(2)
        s2 = a + 1
        dtQ1 = Q2[s2 - 1] - Qmf[s1 - 1, a]
        Qmf[s1 - 1, a] += lr * dtQ1
        dtQ2 = points - Q2[s2 - 1]
        Q2[s2 - 1] += lr * dtQ2
        Qmf[s1 - 1, a] += lam * lr * dtQ2
    return -LL

def fit_one(subdata, seed=0, n_restart=3):
    rng = np.random.default_rng(seed); best = None
    for _ in range(n_restart):
        x0 = [rng.uniform(1, 8), rng.uniform(0.2, 0.8),
              rng.uniform(0.2, 0.8), rng.uniform(0, 1)]
        r = minimize(kool_negll, x0, args=(subdata,),
                     method="L-BFGS-B",
                     bounds=[(0.01, 20), (0.01, 0.99), (0.01, 0.99), (0, 1)],
                     options={"maxiter": 100})
        if best is None or r.fun < best.fun: best = r
    return best.x, best.fun

# ---------- Real-data fits (parallel) -------------------------------------
print(f"\n=== Fitting {len(sub_data)} real subjects (parallel, {N_JOBS} jobs) ===", flush=True)
t0 = time.time()
items = list(sub_data.items())
res = Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(fit_one)(sd, seed=1000+i) for i, (sid, sd) in enumerate(items)
)
rows = []
for (sid, sd), (params, ll) in zip(items, res):
    b, lr, lam, w = params
    rows.append({"subject": sid, "n_trials": len(sd),
                 "b": b, "lr": lr, "lambda": lam, "w_MB": w, "loglik": -ll})
real = pd.DataFrame(rows)
print(f"\nReal fits: {len(real)} subjects in {time.time()-t0:.1f}s")
print(real[["b", "lr", "lambda", "w_MB"]].describe().round(4).to_string())
print(f"  w_MB at 0: {(real['w_MB'] < 0.01).sum()}  at 1: {(real['w_MB'] > 0.99).sum()}")
print(f"  corr(w_MB, lr)={real['w_MB'].corr(real['lr']):.4f}  "
      f"corr(w_MB, lambda)={real['w_MB'].corr(real['lambda']):.4f}")
real.to_csv(OUT / "kool_real_fits.csv", index=False)

# ---------- Recovery (parallel) -------------------------------------------
print(f"\n=== Recovery: 500 synthetic agents (parallel) ===", flush=True)
N_SIM = 500; T_TRIALS = 200
rng = np.random.default_rng(42)
true_b   = rng.uniform(1.0, 8.0, N_SIM)
true_lr  = rng.uniform(0.1, 0.8, N_SIM)
true_lam = rng.uniform(0.1, 0.8, N_SIM)
true_w   = rng.uniform(0.0, 1.0, N_SIM)

def simulate(b, lr, lam, w, n_trials, seed):
    rng = np.random.default_rng(seed)
    Qmf = np.zeros((2, 2)); Q2 = np.zeros(2)
    Tm = [np.full((2,2), 0.5), np.full((2,2), 0.5)]
    Tmchanged = [False, False]
    rows = []
    for t in range(n_trials):
        s1 = int(rng.integers(1, 3))
        Qmb = Tm[s1-1].T @ Q2
        Q = w * Qmb + (1 - w) * Qmf[s1-1, :]
        p = np.exp(b * Q - logsumexp(b * Q))
        a_py = int(rng.choice(2, p=p))
        choice1 = 2*(s1-1) + a_py + 1
        s2 = a_py + 1
        points = float(rng.normal(0.5, 1.5))
        if not Tmchanged[s1-1]:
            Tmchanged[s1-1] = True; Tm[s1-1] = np.eye(2)
        dtQ1 = Q2[s2-1] - Qmf[s1-1, a_py]
        Qmf[s1-1, a_py] += lr * dtQ1
        dtQ2 = points - Q2[s2-1]
        Q2[s2-1] += lr * dtQ2
        Qmf[s1-1, a_py] += lam * lr * dtQ2
        rows.append([s1, 1, 2, 1.0, choice1, 1.0, points, s2, 0,0,0,0, t, 0])
    return np.array(rows, dtype=float)

def rec_one(i):
    sd = simulate(true_b[i], true_lr[i], true_lam[i], true_w[i],
                   T_TRIALS, seed=2000+i)
    return fit_one(sd, seed=3000+i, n_restart=2)[0]

t1 = time.time()
rec_list = Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(rec_one)(i) for i in range(N_SIM))
rec = np.array(rec_list)
print(f"\nRecovery done in {time.time()-t1:.1f}s")

rho_w, p_w = spearmanr(true_w, rec[:, 3])
bias_w = float(np.mean(rec[:, 3] - true_w))
rmse_w = float(np.sqrt(np.mean((rec[:, 3] - true_w)**2)))
rho_lr, _  = spearmanr(true_lr, rec[:, 1])
rho_lam, _ = spearmanr(true_lam, rec[:, 2])
rho_b, _   = spearmanr(true_b, rec[:, 0])

print(f"\n=== Recovery results ===")
print(f"  w_MB:   ρ = {rho_w:.4f}   bias = {bias_w:+.4f}   RMSE = {rmse_w:.4f}")
print(f"  lr:     ρ = {rho_lr:.4f}")
print(f"  lambda: ρ = {rho_lam:.4f}")
print(f"  b:      ρ = {rho_b:.4f}")
print(f"\n  G5 threshold: ρ > 0.6")
print(f"  Verdict: {'PASS' if rho_w > 0.6 else 'FAIL'}")

with open(OUT / "kool_recovery.txt", "w") as f:
    f.write(f"N_SIM={N_SIM}\nT_TRIALS={T_TRIALS}\n")
    f.write(f"rho_w={rho_w:.4f}\np_w={p_w:.3e}\n")
    f.write(f"bias_w={bias_w:.4f}\nrmse_w={rmse_w:.4f}\n")
    f.write(f"rho_lr={rho_lr:.4f}\nrho_lam={rho_lam:.4f}\nrho_b={rho_b:.4f}\n")
    f.write(f"pass={'YES' if rho_w > 0.6 else 'NO'}\n")
pd.DataFrame({
    "true_w": true_w, "rec_w": rec[:, 3],
    "true_b": true_b, "rec_b": rec[:, 0],
    "true_lr": true_lr, "rec_lr": rec[:, 1],
    "true_lam": true_lam, "rec_lam": rec[:, 2],
}).to_csv(OUT / "kool_recovery_scatter.csv", index=False)
with open(OUT / "kool_real_fits.pkl", "wb") as fh: pickle.dump(real, fh)
print(f"\nSaved to {OUT}")
