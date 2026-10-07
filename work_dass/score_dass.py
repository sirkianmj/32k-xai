import pandas as pd
import numpy as np
from pathlib import Path
import time, pickle

SRC = Path.home() / "projects/xai_project/Data_Xai32kProje/XAI_Mental_Health_DASS/data.csv"
OUT = Path.home() / "projects/xai_project/work_dass"

# DASS-42 item -> subscale (standard 14-item mapping)
D42_DEP = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
D42_ANX = [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]
D42_STR = [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]

# DASS-21 subset (7 items per subscale, per Lovibond & Lovibond 1995)
D21_DEP = [3, 5, 10, 13, 16, 17, 21]
D21_ANX = [2, 4, 7, 9, 15, 19, 20]
D21_STR = [1, 6, 8, 11, 12, 14, 18]

A_COLS = [f"Q{i}A" for i in range(1, 43)]
EXTRA = ["uniquenetworklocation"]

t0 = time.time()
df = pd.read_csv(SRC, sep="\t", usecols=A_COLS + EXTRA, low_memory=False)
n_in = len(df)

# Range filter: every A value must be in {1,2,3,4}
mask = np.ones(len(df), dtype=bool)
for c in A_COLS:
    v = df[c].to_numpy()
    mask &= (v >= 1) & (v <= 4) & (v == np.floor(v))
df = df[mask]
n_range = len(df)

# Standard DASS scoring: 1-4 → 0-3
for c in A_COLS:
    df[c] = df[c] - 1

def subscale(items, prefix):
    cols = [f"Q{i}A" for i in items]
    return df[cols].sum(axis=1).rename(prefix)

# DASS-42 subscales (each item range 0-3, 14 items, total 0-42)
d42 = pd.DataFrame({
    "D42_Dep": subscale(D42_DEP, "D"),
    "D42_Anx": subscale(D42_ANX, "A"),
    "D42_Str": subscale(D42_STR, "S"),
})
# DASS-21 subset (7 items each, total 0-21)
d21 = pd.DataFrame({
    "D21_Dep": subscale(D21_DEP, "D"),
    "D21_Anx": subscale(D21_ANX, "A"),
    "D21_Str": subscale(D21_STR, "S"),
})

# Attach uniquenetworklocation for optional filtering
d42["uniquenetworklocation"] = df["uniquenetworklocation"].values
d21["uniquenetworklocation"] = df["uniquenetworklocation"].values

# Normalize to [0,1]
for c in ["D42_Dep", "D42_Anx", "D42_Str"]:
    d42[c + "_n"] = d42[c] / 42.0
for c in ["D21_Dep", "D21_Anx", "D21_Str"]:
    d21[c + "_n"] = d21[c] / 21.0

# Optional cleaner subset: unique network location
d42_clean = d42[d42["uniquenetworklocation"] == 1].copy()
d21_clean = d21[d21["uniquenetworklocation"] == 1].copy()

def cronbach_alpha(items_df):
    items_df = items_df.dropna()
    k = items_df.shape[1]
    if k < 2 or len(items_df) < 2:
        return float("nan")
    var_items = items_df.var(axis=0, ddof=1).sum()
    var_total = items_df.sum(axis=1).var(ddof=1)
    if var_total == 0:
        return float("nan")
    return (k / (k - 1)) * (1 - var_items / var_total)

alpha42_dep = cronbach_alpha(df[[f"Q{i}A" for i in D42_DEP]])
alpha42_anx = cronbach_alpha(df[[f"Q{i}A" for i in D42_ANX]])
alpha42_str = cronbach_alpha(df[[f"Q{i}A" for i in D42_STR]])
alpha21_dep = cronbach_alpha(df[[f"Q{i}A" for i in D21_DEP]])
alpha21_anx = cronbach_alpha(df[[f"Q{i}A" for i in D21_ANX]])
alpha21_str = cronbach_alpha(df[[f"Q{i}A" for i in D21_STR]])

print(f"Rows read:                  {n_in}")
print(f"After [1,4]-integer filter: {n_range}")
print(f"After uniquenetworkloc==1:  {len(d42_clean)}")
print(f"Elapsed:                    {time.time()-t0:.1f}s")
print()
print("=== Cronbach's alpha ===")
print(f"  DASS-42  Dep={alpha42_dep:.4f}  Anx={alpha42_anx:.4f}  Str={alpha42_str:.4f}")
print(f"  DASS-21  Dep={alpha21_dep:.4f}  Anx={alpha21_anx:.4f}  Str={alpha21_str:.4f}")
print()
print("=== DASS-42 subscales (raw 0-42) ===")
print(d42[["D42_Dep","D42_Anx","D42_Str"]].describe().round(3).to_string())
print()
print("=== DASS-21 subset (raw 0-21) ===")
print(d21[["D21_Dep","D21_Anx","D21_Str"]].describe().round(3).to_string())
print()
print("=== Correlation, DASS-42 subscales ===")
print(d42[["D42_Dep","D42_Anx","D42_Str"]].corr().round(3).to_string())
print()
print("=== Correlation, DASS-21 subscales ===")
print(d21[["D21_Dep","D21_Anx","D21_Str"]].corr().round(3).to_string())
print()

assert d42[["D42_Dep","D42_Anx","D42_Str"]].min().min() >= 0, "Range violation (low)"
assert d42[["D42_Dep","D42_Anx","D42_Str"]].max().max() <= 42, "Range violation (high)"
assert d21[["D21_Dep","D21_Anx","D21_Str"]].min().min() >= 0, "Range violation (low)"
assert d21[["D21_Dep","D21_Anx","D21_Str"]].max().max() <= 21, "Range violation (high)"
print("Range checks OK")

d42.to_csv(OUT / "dass42_subscales.csv", index=False)
d21.to_csv(OUT / "dass21_subscales.csv", index=False)
with open(OUT / "dass42_subscales.pkl", "wb") as fh: pickle.dump(d42, fh, protocol=4)
with open(OUT / "dass21_subscales.pkl", "wb") as fh: pickle.dump(d21, fh, protocol=4)
d42.sample(n=min(50_000, len(d42)), random_state=42).to_csv(
    OUT / "dass42_sample50k.csv", index=False)
print(f"Saved: dass42_subscales.csv + dass21_subscales.csv + .pkl + 50k sample")
