import pandas as pd
import numpy as np
from pathlib import Path
import time, pickle

SRC = Path.home() / "projects/xai_project/Data_Xai32kProje/SD3/data.csv"
OUT = Path.home() / "projects/xai_project/work_sd3"

M_ITEMS = [f"M{i}" for i in range(1, 10)]
N_ITEMS = [f"N{i}" for i in range(1, 10)]
P_ITEMS = [f"P{i}" for i in range(1, 10)]
ITEMS = M_ITEMS + N_ITEMS + P_ITEMS

# Canonical SD3 reverse-keyed items (Paulhus & Jones 2011)
REV = {"N2", "N6", "N8", "P2", "P7"}

EXTRA = ["country", "source"]

t0 = time.time()
df = pd.read_csv(SRC, sep="\t", usecols=ITEMS + EXTRA, low_memory=False)
n_in = len(df)

# Range filter: every item must be integer in {1,2,3,4,5}
mask = np.ones(len(df), dtype=bool)
for c in ITEMS:
    v = df[c].to_numpy()
    mask &= (v >= 1) & (v <= 5) & (v == np.floor(v))
df = df[mask].reset_index(drop=True)
n_range = len(df)

# Preserve raw items for CFA (before reverse-keying)
raw_items = df[ITEMS].copy()
raw_items.to_csv(OUT / "sd3_items_raw.csv", index=False)

# Reverse-keyed version for CFA
rev_items = df[ITEMS].copy()
for c in REV:
    rev_items[c] = 6 - rev_items[c]
rev_items.to_csv(OUT / "sd3_items_revkeyed.csv", index=False)

# Subscale sums on reverse-keyed data
subs = pd.DataFrame({
    "Mach": rev_items[M_ITEMS].sum(axis=1),
    "Narc": rev_items[N_ITEMS].sum(axis=1),
    "Psyc": rev_items[P_ITEMS].sum(axis=1),
})
# Range: 9 items × [1,5] → [9, 45]
subs["Mach_n"] = (subs["Mach"] - 9) / 36.0
subs["Narc_n"] = (subs["Narc"] - 9) / 36.0
subs["Psyc_n"] = (subs["Psyc"] - 9) / 36.0
subs["country"] = df["country"].values
subs["source"]  = df["source"].values
subs.to_csv(OUT / "sd3_subscales.csv", index=False)
with open(OUT / "sd3_subscales.pkl", "wb") as fh:
    pickle.dump(subs, fh, protocol=4)

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

alpha_m = cronbach_alpha(rev_items[M_ITEMS])
alpha_n = cronbach_alpha(rev_items[N_ITEMS])
alpha_p = cronbach_alpha(rev_items[P_ITEMS])

print(f"Rows read:                  {n_in}")
print(f"After [1,5]-integer filter: {n_range}")
print(f"Elapsed:                    {time.time()-t0:.1f}s")
print()
print("=== Cronbach's alpha (reverse-keyed) ===")
print(f"  Machiavellianism  {alpha_m:.4f}   (published ~0.71-0.79)")
print(f"  Narcissism        {alpha_n:.4f}   (published ~0.71-0.77)")
print(f"  Psychopathy       {alpha_p:.4f}   (published ~0.63-0.75)")
print()
print("=== Subscales raw (range 9-45) ===")
print(subs[["Mach","Narc","Psyc"]].describe().round(3).to_string())
print()
print("=== Subscales normalized (range 0-1) ===")
print(subs[["Mach_n","Narc_n","Psyc_n"]].describe().round(4).to_string())
print()
print("=== Subscale correlations ===")
print(subs[["Mach","Narc","Psyc"]].corr().round(3).to_string())
print()

assert subs[["Mach","Narc","Psyc"]].min().min() >= 9,  "Range violation (low)"
assert subs[["Mach","Narc","Psyc"]].max().max() <= 45, "Range violation (high)"
print("Range checks OK")

print(f"Saved: sd3_items_raw.csv + sd3_items_revkeyed.csv + sd3_subscales.csv/.pkl")
