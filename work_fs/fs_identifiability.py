import numpy as np, time, warnings
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

OUT = Path.home() / "projects/xai_project/work_fs"

def sigmoid(x):
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0/(1.0+np.exp(-x[pos]))
    ex = np.exp(x[~pos]); out[~pos] = ex/(1.0+ex)
    return out

def neg_ll(p, offer, endow, acc):
    a, t = p
    U = offer - a*(endow - 2.0*offer)
    pr = np.clip(sigmoid(t*U), 1e-9, 1-1e-9)
    return -np.sum(acc*np.log(pr) + (1-acc)*np.log(1-pr))

def fit(offer, endow, acc, seed, nr=3):
    rng = np.random.default_rng(seed); best = None
    for _ in range(nr):
        x0 = [rng.uniform(0.02, 0.5), rng.uniform(2, 20)]
        r = minimize(neg_ll, x0, args=(offer, endow, acc),
                     method="L-BFGS-B", bounds=[(0, 10), (1e-3, 50)],
                     options={"maxiter": 100})
        if best is None or r.fun < best.fun: best = r
    return best.x

OFF_16 = [0.0, 1.0, 2.0, 4.0, 8.0]
OFF_32 = [0.0, 2.0, 3.0, 8.0, 16.0]

def sim(a, t, n, seed):
    r = np.random.default_rng(seed)
    E = r.choice([16.0, 32.0], size=n)
    offers = np.where(E==16, r.choice(OFF_16, size=n), r.choice(OFF_32, size=n))
    U = offers - a*(E - 2*offers)
    return offers, E, (r.random(n) < sigmoid(t*U)).astype(int)

def recovery(true_alpha, true_tau, n_per=500, label="", seed=42):
    rng = np.random.default_rng(seed)
    rec = np.full(n_per, np.nan)
    t0 = time.time()
    for i in range(n_per):
        n = int(rng.integers(40, 120))
        o, e, a = sim(true_alpha[i], true_tau[i], n, seed=2000+i)
        if a.sum() == 0 or a.sum() == len(a): continue
        rec[i] = fit(o, e, a, seed=3000+i, nr=2)[0]
    m = ~np.isnan(rec)
    rho, _ = spearmanr(true_alpha[m], rec[m])
    bias = float(np.mean(rec[m] - true_alpha[m]))
    rmse = float(np.sqrt(np.mean((rec[m] - true_alpha[m])**2)))
    print(f"  {label:32s}  N={m.sum():3d}  rho={rho:.4f}  bias={bias:+.4f}  rmse={rmse:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    return rho, bias, rmse, int(m.sum())

print("=== Recovery by true-alpha regime (same design as UGR, 500 sims each) ===", flush=True)
rng = np.random.default_rng(7)
N = 500

regimes = [
    ("narrow_low  [0.02,0.30]",  rng.uniform(0.02, 0.30, N), rng.uniform( 5, 30, N), 100),
    ("narrow_low  [0.02,0.50]",  rng.uniform(0.02, 0.50, N), rng.uniform( 5, 30, N), 101),
    ("lowmid      [0.20,0.80]",  rng.uniform(0.20, 0.80, N), rng.uniform( 5, 30, N), 102),
    ("mid         [0.50,1.50]",  rng.uniform(0.50, 1.50, N), rng.uniform( 5, 30, N), 103),
    ("wide        [0.02,3.00]",  rng.uniform(0.02, 3.00, N), rng.uniform( 5, 30, N), 104),
    ("log-uniform [0.02,3.00]",  10**rng.uniform(np.log10(0.02), np.log10(3.0), N), rng.uniform(5, 30, N), 105),
]

results = {}
for label, ta, tt, sd in regimes:
    rho, bias, rmse, nv = recovery(ta, tt, N, label, seed=sd)
    results[label] = dict(rho=rho, bias=bias, rmse=rmse, n=nv)

with open(OUT / "fs_identifiability.txt", "w") as f:
    for k, v in results.items():
        f.write(f"{k:32s}  rho={v['rho']:.4f}  bias={v['bias']:+.4f}  rmse={v['rmse']:.4f}  N={v['n']}\n")
print(f"\nSaved to {OUT}/fs_identifiability.txt")
