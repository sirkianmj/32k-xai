import pandas as pd
import numpy as np
from pathlib import Path
import time, pickle

SRC = Path.home() / "projects/xai_project/Data_Xai32kProje/XAI_Delay_Discounting_Data/Delay Discounting Task_March 21, 2018.csv"
OUT = Path.home() / "projects/xai_project/work_dd"

t0 = time.time()
df = pd.read_csv(SRC, low_memory=False)
n_in = len(df)
df = df[df["Distribution Channel"] != "preview"].copy().reset_index(drop=True)
n_anon = len(df)

# Positional indices for the 31 choice trials
choice_idx = list(range(12, 163, 5))
assert len(choice_idx) == 31, f"expected 31 choice columns, got {len(choice_idx)}"

def classify(v):
    if not isinstance(v, str):
        return None
    return 1 if "/ 2" in v else 0

# Per-trial diagnostic
print("=== per-trial non-null + SS/LL split ===")
for j, i in enumerate(choice_idx):
    col = df.iloc[:, i]
    n_str = col.apply(lambda x: isinstance(x, str)).sum()
    n_ss = col.apply(lambda x: isinstance(x, str) and "/ 2" in x).sum()
    n_ll = n_str - n_ss
    print(f"  trial {j+1:2d}  idx={i:3d}  answered={n_str:4d}  SS={n_ss:4d}  LL={n_ll:4d}")

# Per-subject counts (positional)
n_rows = len(df)
ss_counts = np.zeros(n_rows, dtype=int)
ll_counts = np.zeros(n_rows, dtype=int)
answered_counts = np.zeros(n_rows, dtype=int)
for i in choice_idx:
    col = df.iloc[:, i].values
    for k in range(n_rows):
        c = classify(col[k])
        if c == 1:
            ss_counts[k] += 1; answered_counts[k] += 1
        elif c == 0:
            ll_counts[k] += 1; answered_counts[k] += 1

out = pd.DataFrame({
    "response_id":    df["Response ID"].values,
    "subject_number": df["Subject Number:"].values,
    "k":              pd.to_numeric(df["k value"], errors="coerce").values,
    "ED50":           pd.to_numeric(df["ED50"], errors="coerce").values,
    "duration_s":     pd.to_numeric(df["Duration (in seconds)"], errors="coerce").values,
    "n_SS":           ss_counts,
    "n_LL":           ll_counts,
    "n_answered":     answered_counts,
})

mask = np.isfinite(out["k"]) & (out["k"] > 0)
out = out[mask].reset_index(drop=True)
out["log_k"] = np.log(out["k"])

print()
print(f"Rows read:                 {n_in}")
print(f"After preview drop:        {n_anon}")
print(f"After k>0 finite filter:   {len(out)}")
print(f"Elapsed:                   {time.time()-t0:.1f}s")
print()
print("=== k distribution ===")
print(out["k"].describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).round(6).to_string())
print()
print("=== log k distribution (for copula marginal) ===")
print(out["log_k"].describe().round(4).to_string())
print()
print("=== n_answered per subject ===")
print(out["n_answered"].describe().round(2).to_string())
print()
print("=== n_SS per subject ===")
print(out["n_SS"].describe().round(2).to_string())
print()

full = out[out["n_answered"] == 31]
partial = out[(out["n_answered"] >= 20) & (out["n_answered"] < 31)]
print(f"subjects with all 31 trials:  {len(full)}")
print(f"subjects with 20-30 trials:   {len(partial)}")
print(f"subjects with <20 trials:     {len(out) - len(full) - len(partial)}")
print()

if len(full) >= 10:
    r_full = full["k"].corr(full["n_SS"])
    print(f"corr(k, n_SS) on ALL-31 subjects (n={len(full)}) = {r_full:.4f}")
if len(out) >= 10:
    r_all = out["k"].corr(out["n_SS"])
    print(f"corr(k, n_SS) on all subjects      (n={len(out)}) = {r_all:.4f}")
if len(partial) >= 10:
    # normalize by trials answered
    partial_r = partial.assign(frac_ss=partial["n_SS"]/partial["n_answered"])
    r_part = partial_r["k"].corr(partial_r["frac_ss"])
    print(f"corr(k, SS_fraction) on partial    (n={len(partial)}) = {r_part:.4f}")
print()

assert (out["k"] > 0).all(), "k must be positive"
print("Range check: all k > 0  OK")

out.to_csv(OUT / "dd_k_subjects.csv", index=False)
with open(OUT / "dd_k_subjects.pkl", "wb") as fh:
    pickle.dump(out, fh, protocol=4)
print(f"Saved: dd_k_subjects.csv + .pkl")
