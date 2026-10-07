"""
G5 — Kool novel two-step paradigm (w_MB recovery) — CORRECTED likelihood
"""
import numpy as np, pandas as pd, pickle, time, warnings
from pathlib import Path
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from numba import njit
warnings.filterwarnings("ignore")

ROOT  = Path.home() / "projects/xai_project"
TRADE = ROOT / "tradeoffs/data/novel paradigm/data.mat"
OUT   = ROOT / "work_kool"
N_JOBS = -1

# =========================================================================
# FIXED LIKELIHOOD: log_softmax(b*Q)[a] = b*Q[a] - logsumexp(b*Q)
# =========================================================================
@njit(cache=True, fastmath=True)
def kool_negll_numba(params, data):
    b, lr, lam, w = params
    if b <= 0.01 or b >= 20.0: return 1e10
    if lr <= 0.01 or lr >= 0.99: return 1e10
    if lam <= 0.01 or lam >= 0.99: return 1e10
    if w < 0.0 or w > 1.0: return 1e10

    Qmf = np.zeros((2, 2))
    Q2  = np.zeros(2)
    Tm = np.array([[0.5, 0.5, 0.5, 0.5],
                   [0.5, 0.5, 0.5, 0.5]])
    Tmchanged = np.zeros(2, dtype=np.bool_)
    R = np.zeros(2)

    LL = 0.0
    N = data.shape[0]
    for t in range(N):
        s1     = int(data[t, 0])
        stim_l = int(data[t, 1])
        rt1    = data[t, 3]
        c1     = int(data[t, 4])
        rt2    = data[t, 5]
        points = data[t, 6]

        if rt1 == -1.0 or rt2 == -1.0:
            continue

        if stim_l == 2 or stim_l == 4:
            tmp = R[0]; R[0] = R[1]; R[1] = tmp

        a = (c1 - 1) % 2

        q_mb_0 = Tm[s1-1, 0]*Q2[0] + Tm[s1-1, 2]*Q2[1]
        q_mb_1 = Tm[s1-1, 1]*Q2[0] + Tm[s1-1, 3]*Q2[1]

        q0 = w * q_mb_0 + (1.0 - w) * Qmf[s1-1, 0]
        q1 = w * q_mb_1 + (1.0 - w) * Qmf[s1-1, 1]

        # ---- FIX: multiply Q by b BEFORE the logsumexp ----
        bq0 = b * q0
        bq1 = b * q1
        qa = bq0 if a == 0 else bq1
        m = bq0 if bq0 > bq1 else bq1
        logsumexp_ = m + np.log(np.exp(bq0 - m) + np.exp(bq1 - m))
        LL += qa - logsumexp_

        if not Tmchanged[s1-1]:
            Tmchanged[s1-1] = True
            Tm[s1-1] = np.array([1.0, 0.0, 0.0, 1.0])

        s2i = a
        dtQ1 = Q2[s2i] - Qmf[s1-1, a]
        Qmf[s1-1, a] += lr * dtQ1
        dtQ2 = points - Q2[s2i]
        Q2[s2i] += lr * dtQ2
        Qmf[s1-1, a] += lam * lr * dtQ2

    return -LL

_ = kool_negll_numba(np.array([3.0, 0.5, 0.5, 0.5]),
                     np.zeros((2, 14), dtype=np.float64))

def negll_scipy(x, data):
    return kool_negll_numba(np.asarray(x, dtype=np.float64), data)

def fit_one(subdata, seed=0, n_restart=3):
    rng = np.random.default_rng(seed); best = None
    for _ in range(n_restart):
        x0 = np.array([rng.uniform(0.5, 6), rng.uniform(0.1, 0.9),
                       rng.uniform(0.1, 0.9), rng.uniform(0, 1)])
        r = minimize(negll_scipy, x0, args=(subdata,),
                     method="L-BFGS-B",
                     bounds=[(0.01, 20), (0.01, 0.99), (0.01, 0.99), (0, 1)],
                     options={"maxiter": 200})
        if best is None or r.fun < best.fun: best = r
    return best.x, best.fun

# =========================================================================
# Load
# =========================================================================
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
    sub_data[sid] = arr[keep].astype(np.float64)
print(f"  {len(sub_data)} subjects loaded", flush=True)

# Empirical reward distribution (for simulator)
all_points = np.concatenate([sd[:, 6] for sd in sub_data.values()])
print(f"  reward mean={all_points.mean():.3f}  std={all_points.std():.3f}")
unique_pts, counts = np.unique(all_points, return_counts=True)
probs = counts / counts.sum()
print(f"  unique rewards: {unique_pts.tolist()}")

# =========================================================================
# Real fits
# =========================================================================
print(f"\n=== Fitting {len(sub_data)} real subjects ===", flush=True)
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
print(f"  b at 20:  {(real['b'] > 19.99).sum()}")
real.to_csv(OUT / "kool_real_fits.csv", index=False)

# =========================================================================
# Recovery — simulator uses empirical reward distribution
# =========================================================================
print(f"\n=== Recovery: 500 synthetic agents ===", flush=True)
N_SIM = 500; T_TRIALS = 200
rng = np.random.default_rng(42)
true_b   = rng.uniform(0.5, 8.0, N_SIM)
true_lr  = rng.uniform(0.1, 0.9, N_SIM)
true_lam = rng.uniform(0.1, 0.9, N_SIM)
true_w   = rng.uniform(0.0, 1.0, N_SIM)

# Empirical reward distribution for simulation
pts = np.asarray(unique_pts, dtype=np.float64)
prb = np.asarray(probs, dtype=np.float64)

@njit(cache=True, fastmath=True)
def simulate_numba(b, lr, lam, w, n_trials, seed, pts, prb):
    np.random.seed(seed)
    Qmf = np.zeros((2, 2))
    Q2  = np.zeros(2)
    Tm = np.array([[0.5, 0.5, 0.5, 0.5],
                   [0.5, 0.5, 0.5, 0.5]])
    Tmchanged = np.zeros(2, dtype=np.bool_)
    rows = np.zeros((n_trials, 14))
    for t in range(n_trials):
        s1 = np.random.randint(1, 3)

        q_mb_0 = Tm[s1-1, 0]*Q2[0] + Tm[s1-1, 2]*Q2[1]
        q_mb_1 = Tm[s1-1, 1]*Q2[0] + Tm[s1-1, 3]*Q2[1]
        q0 = w * q_mb_0 + (1.0 - w) * Qmf[s1-1, 0]
        q1 = w * q_mb_1 + (1.0 - w) * Qmf[s1-1, 1]

        m = q0 if q0 > q1 else q1
        e0 = np.exp(b * (q0 - m))
        e1 = np.exp(b * (q1 - m))
        p0 = e0 / (e0 + e1)
        a_py = 0 if np.random.random() < p0 else 1

        choice1 = 2*(s1-1) + a_py + 1
        s2 = a_py + 1

        # Empirical reward draw
        u = np.random.random()
        c = 0.0
        for k in range(len(prb)):
            c += prb[k]
            if u < c:
                points = pts[k]; break

        if not Tmchanged[s1-1]:
            Tmchanged[s1-1] = True
            Tm[s1-1] = np.array([1.0, 0.0, 0.0, 1.0])

        dtQ1 = Q2[s2-1] - Qmf[s1-1, a_py]
        Qmf[s1-1, a_py] += lr * dtQ1
        dtQ2 = points - Q2[s2-1]
        Q2[s2-1] += lr * dtQ2
        Qmf[s1-1, a_py] += lam * lr * dtQ2

        rows[t, 0] = s1
        rows[t, 1] = 1
        rows[t, 3] = 1.0
        rows[t, 4] = choice1
        rows[t, 5] = 1.0
        rows[t, 6] = points
    return rows

_ = simulate_numba(3.0, 0.5, 0.5, 0.5, 5, 1, pts, prb)

def rec_one(i):
    sd = simulate_numba(true_b[i], true_lr[i], true_lam[i], true_w[i],
                        T_TRIALS, 2000+i, pts, prb)
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
print(f"  w_MB:   rho = {rho_w:.4f}   bias = {bias_w:+.4f}   RMSE = {rmse_w:.4f}")
print(f"  lr:     rho = {rho_lr:.4f}")
print(f"  lambda: rho = {rho_lam:.4f}")
print(f"  b:      rho = {rho_b:.4f}")
print(f"\n  G5 threshold: rho > 0.6")
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
