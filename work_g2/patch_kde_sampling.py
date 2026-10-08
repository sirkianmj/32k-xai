# patch_kde_sampling.py — corrected KDE resample for kde_anchored columns.
# Fix from prior attempt: draw exactly n samples (not 1.5n truncated).
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import gaussian_kde

ROOT = Path.home() / "projects/xai_project"
OUT  = ROOT / "work_g2"
SEED = 20261007

ts = pd.read_csv(ROOT / "work_n500/w_mb_real.csv")
dd = pd.read_csv(ROOT / "work_dd/dd_k_subjects.csv")
ss = pd.read_csv(ROOT / "work_ssrt/ssrt_subjects.csv")

def logit(x, eps=1e-6):
    x = np.clip(x, eps, 1-eps); return np.log(x/(1-x))
def inv_logit(z):
    return 1.0/(1.0+np.exp(-z))

# (col, src, transform)
SPEC = [
    ("alpha_R",     ts["alpha"].to_numpy(),         "logit"),
    ("beta_choice", ts["beta"].to_numpy(),          "log"),
    ("phi_per",     ts["persev"].to_numpy(),        "id"),
    ("w_MB",        ts["w"].to_numpy(),             "logit"),
    ("a_inhib",     ss["ssrt"].dropna().to_numpy(), "log"),
    ("k_DD",        dd["log_k"].to_numpy(),         "id"),
]

def fwd(kind, x):
    if kind == "logit": return logit(x)
    if kind == "log":   return np.log(np.clip(x, 1e-12, None))
    return np.asarray(x, dtype=np.float64)

def inv(kind, z):
    if kind == "logit": return inv_logit(z)
    if kind == "log":   return np.exp(z)
    return z

df = pd.read_pickle(OUT / "synthetic_32k.pkl")
print(f"Before: shape={df.shape}")
rng = np.random.default_rng(SEED + 1)

for col, src, kind in SPEC:
    cur = df[col].to_numpy().astype(np.float64)
    n = len(cur)
    uniq_before = len(np.unique(cur))
    src_min, src_max = float(np.min(src)), float(np.max(src))

    z_src = fwd(kind, src)
    kde = gaussian_kde(z_src, bw_method="silverman")

    # Draw exactly n samples. No truncation.
    seed_i = int(rng.integers(0, 2**31))
    z_draw = kde.resample(n, seed=seed_i).flatten()
    z_draw = np.sort(z_draw)                     # ascending
    new_vals = inv(kind, z_draw)                 # back to native space

    # Rank-match: place sorted new values at positions sorted by current values
    order = np.argsort(np.argsort(cur))          # 0..n-1 ranks of current
    new_col = np.empty(n, dtype=np.float64)
    new_col[order] = new_vals

    # Range check: report how KDE-drawn range compares to source range
    lo_new, hi_new = float(new_col.min()), float(new_col.max())
    df[col] = new_col.astype(np.float32)
    uniq_after = len(np.unique(new_col))

    print(f"  {col:14s} src_n={len(src):5d}  uniq_before={uniq_before:6d}  "
          f"uniq_after={uniq_after:6d}")
    print(f"                 src_range=[{src_min:+.4f},{src_max:+.4f}]  "
          f"new_range=[{lo_new:+.4f},{hi_new:+.4f}]")

df.to_pickle(OUT / "synthetic_32k.pkl")
print(f"After:  shape={df.shape}")
print(f"Saved:  {OUT / 'synthetic_32k.pkl'}")
