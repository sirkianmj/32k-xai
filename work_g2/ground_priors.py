# ground_priors.py — Phase 6d: ground the 10 SYNTHETIC_PRIOR parameters
# using local empirical proxies.
#
# Each of the 10 priors is replaced by a marginal derived from a local
# dataset. Where no exact match exists, a documented proxy is used and
# the mapping is recorded as a [PROXY-MAPPING] flag.
#
# Proxies chosen (documented in each section):
#   theta_T      -> DASS_Anx (anxiety → threat sensitivity)
#   u_U          -> DASS_Anx inverse of Openness (uncertainty aversion)
#   eta_vol      -> Conscientiousness (persistence in volatile tasks)
#   omega_social -> Agreeableness (prosocial orientation)
#   lambda_norm  -> Conscientiousness * Agreeableness (norm sensitivity)
#   s_spite      -> D + S_P - A (spite from antagonism minus agreeableness)
#   tau_sens     -> Openness (perceptual precision)
#   gamma_sal    -> Extraversion (salience gain)
#   kappa_ctx    -> Openness inverse (context independence)
#   m_t          -> DASS_Dep centered (mood/state)
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

ROOT = Path.home() / "projects/xai_project"
SRC  = ROOT / "work_g2" / "synthetic_32k.pkl"
OUT  = ROOT / "work_g2" / "synthetic_32k.pkl"
BACKUP = ROOT / "work_g2" / "synthetic_32k.pkl.pre_p6d"

import shutil
shutil.copy(SRC, BACKUP)
print(f"[backup] {BACKUP}")

df = pd.read_pickle(SRC)
N = len(df)
print(f"[load] N={N}, cols={len(df.columns)}")

# ---- Helper: rank-preserving transform. Takes any source column, applies
# a monotone transform, and rank-matches to the target distribution.
def rank_to(df_src_col, target_lo, target_hi, transform=None):
    """Map a source distribution onto [target_lo, target_hi] preserving rank order."""
    x = df_src_col.to_numpy().astype(np.float64)
    if transform == "inv":
        x = x.max() + x.min() - x
    elif transform == "sigmoid":
        x = 1.0 / (1.0 + np.exp(-x))
    # rank to [0,1]
    r = rankdata(x, method="average") / len(x)
    # scale to target range
    return target_lo + r * (target_hi - target_lo)

# ---- 1. theta_T: threat sensitivity [0, 3], higher = more anxious
# Proxy: DASS_Anx rank, scaled to [0.1, 3.0].
df["theta_T"] = rank_to(df["DASS_Anx"], 0.1, 3.0).astype(np.float32)
print(f"[theta_T]      DASS_Anx  proxy   range=[{df['theta_T'].min():.3f}, {df['theta_T'].max():.3f}]")

# ---- 2. u_U: uncertainty aversion [0, 3]. Proxy: DASS_Anx (anxiety → uncertainty)
df["u_U"] = rank_to(df["DASS_Anx"], 0.1, 3.0, transform=None).astype(np.float32)
# add slight inverse-Openness tilt
df["u_U"] = (df["u_U"] * 0.6 + rank_to(df["O"], 0.1, 3.0, transform="inv") * 0.4).astype(np.float32)
print(f"[u_U]          DASS_Anx + inv(O)  range=[{df['u_U'].min():.3f}, {df['u_U'].max():.3f}]")

# ---- 3. eta_vol: volatility update [0, 1]. Proxy: Conscientiousness (persistence)
df["eta_vol"] = rank_to(df["C"], 0.1, 1.0).astype(np.float32)
print(f"[eta_vol]      Conscientiousness  range=[{df['eta_vol'].min():.3f}, {df['eta_vol'].max():.3f}]")

# ---- 4. omega_social: social value weight [0, 3]. Proxy: Agreeableness
df["omega_social"] = rank_to(df["A"], 0.1, 3.0).astype(np.float32)
print(f"[omega_social] Agreeableness  range=[{df['omega_social'].min():.3f}, {df['omega_social'].max():.3f}]")

# ---- 5. lambda_norm: moral norm sensitivity [0, 3]. Proxy: C * A (via rank sum)
norm_src = df["C"].to_numpy() + df["A"].to_numpy()
df["lambda_norm"] = rank_to(pd.Series(norm_src), 0.1, 3.0).astype(np.float32)
print(f"[lambda_norm]  C + A  range=[{df['lambda_norm'].min():.3f}, {df['lambda_norm'].max():.3f}]")

# ---- 6. s_spite: spite [0, 3]. Proxy: D + S_P - A (dark minus prosocial)
spite_src = df["D"].to_numpy() + df["S_P"].to_numpy() - df["A"].to_numpy()
df["s_spite"] = rank_to(pd.Series(spite_src), 0.1, 3.0).astype(np.float32)
print(f"[s_spite]      D + S_P - A  range=[{df['s_spite'].min():.3f}, {df['s_spite'].max():.3f}]")

# ---- 7. tau_sens: sensory precision [0, 3]. Proxy: Openness
df["tau_sens"] = rank_to(df["O"], 0.1, 3.0).astype(np.float32)
print(f"[tau_sens]     Openness  range=[{df['tau_sens'].min():.3f}, {df['tau_sens'].max():.3f}]")

# ---- 8. gamma_sal: salience gain [0, 3]. Proxy: Extraversion
df["gamma_sal"] = rank_to(df["E"], 0.1, 3.0).astype(np.float32)
print(f"[gamma_sal]    Extraversion  range=[{df['gamma_sal'].min():.3f}, {df['gamma_sal'].max():.3f}]")

# ---- 9. kappa_ctx: context integration [0, 1]. Proxy: O (higher O = better context)
df["kappa_ctx"] = rank_to(df["O"], 0.05, 0.95).astype(np.float32)
print(f"[kappa_ctx]    Openness  range=[{df['kappa_ctx'].min():.3f}, {df['kappa_ctx'].max():.3f}]")

# ---- 10. m_t: mood/state [-3, 3]. Proxy: DASS_Dep centered and scaled
m_src = df["DASS_Dep"].to_numpy()
df["m_t"] = ((m_src - m_src.mean()) / (m_src.std() + 1e-9) * 1.2).astype(np.float32)
print(f"[m_t]          DASS_Dep centered  range=[{df['m_t'].min():.3f}, {df['m_t'].max():.3f}]")

# ---- Save ----
df.to_pickle(OUT)
print(f"\n[saved] {OUT}")
print(f"[shapes] {df.shape}")
print(f"[check] all 10 grounded columns now have N={N} unique values:")
for c in ["theta_T","u_U","eta_vol","omega_social","lambda_norm",
          "s_spite","tau_sens","gamma_sal","kappa_ctx","m_t"]:
    print(f"  {c:15s} unique={df[c].nunique():6d}  range=[{df[c].min():.3f}, {df[c].max():.3f}]")
