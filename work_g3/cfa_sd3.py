import pandas as pd, numpy as np, time, warnings
from pathlib import Path
import semopy
warnings.filterwarnings("ignore")

SRC = Path.home() / "projects/xai_project/work_sd3/sd3_items_revkeyed.csv"
OUT = Path.home() / "projects/xai_project/work_g3"

M = [f"M{i}" for i in range(1,10)]
N = [f"N{i}" for i in range(1,10)]
P = [f"P{i}" for i in range(1,10)]
ALL = M + N + P

df = pd.read_csv(SRC)[ALL].dropna().reset_index(drop=True)
print(f"Input: {df.shape[0]} subjects x {df.shape[1]} items", flush=True)

model_A = "\n".join([f"M =~ {' + '.join(M)}",
                     f"N =~ {' + '.join(N)}",
                     f"P =~ {' + '.join(P)}"])
model_B = f"G =~ {' + '.join(ALL)}"
model_C = "\n".join([f"G =~ {' + '.join(ALL)}",
                     f"SM =~ {' + '.join(M)}",
                     f"SN =~ {' + '.join(N)}",
                     f"SP =~ {' + '.join(P)}",
                     "G ~~ 0*SM","G ~~ 0*SN","G ~~ 0*SP",
                     "SM ~~ 0*SN","SM ~~ 0*SP","SN ~~ 0*SP"])

def fit_and_report(desc, model):
    t0 = time.time()
    try:
        m = semopy.Model(model)
        m.fit(df)
    except Exception as e:
        print(f"\n=== {desc}: FIT FAILED — {type(e).__name__}: {e} ===", flush=True)
        return None, None
    try:
        stats_df = semopy.calc_stats(m)       # rows = models, cols = stats
        row = stats_df.iloc[0]
        print(f"\n=== {desc} ({time.time()-t0:.1f}s) ===", flush=True)
        for k in ["DoF","chi2","chi2 p-value","CFI","TLI","RMSEA","GFI","AGFI","NFI","AIC","BIC","LogLik"]:
            if k in row.index:
                v = row[k]
                if isinstance(v, (int, float, np.floating)):
                    print(f"  {k:15s} {v:.4f}")
                else:
                    print(f"  {k:15s} {v}")
        return m, row
    except Exception as e:
        print(f"\n=== {desc}: STATS FAILED — {type(e).__name__}: {e} ===", flush=True)
        return m, None

mA, sA = fit_and_report("A: three correlated factors", model_A)
mB, sB = fit_and_report("B: unidimensional", model_B)
mC, sC = fit_and_report("C: bifactor", model_C)

print("\n=== Comparison ===")
rows = []
for name, s in [("A:3-correlated", sA), ("B:unidimensional", sB), ("C:bifactor", sC)]:
    if s is None: continue
    rows.append({"model": name,
                 "chi2": float(s.get("chi2", np.nan)),
                 "CFI": float(s.get("CFI", np.nan)),
                 "TLI": float(s.get("TLI", np.nan)),
                 "RMSEA": float(s.get("RMSEA", np.nan)),
                 "AIC": float(s.get("AIC", np.nan)),
                 "BIC": float(s.get("BIC", np.nan))})
comp = pd.DataFrame(rows)
print(comp.round(4).to_string(index=False))
comp.to_csv(OUT / "sd3_cfa_comparison.csv", index=False)

if mC is not None:
    try:
        loadings = semopy.inspect(mC)
        loadings.to_csv(OUT / "sd3_bifactor_loadings.csv", index=False)
        print(f"\nBifactor loadings saved: {loadings.shape}")
    except Exception as e:
        print(f"\nBifactor loadings save failed: {e}")
