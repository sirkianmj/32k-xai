import pandas as pd, numpy as np, time, warnings
from pathlib import Path
import semopy
warnings.filterwarnings("ignore")

OUT = Path.home() / "projects/xai_project/work_g4"
df = pd.read_csv(OUT / "dass42_items.csv")
ALL = list(df.columns)
print(f"Input: {df.shape[0]} subjects x {df.shape[1]} items", flush=True)

# DASS-42 subscale item mapping (standard)
DEP = [3,5,10,13,16,17,21,24,26,31,34,37,38,42]
ANX = [2,4,7,9,15,19,20,23,25,28,30,36,40,41]
STR = [1,6,8,11,12,14,18,22,27,29,32,33,35,39]
DEP_C = [f"Q{i}A" for i in DEP]
ANX_C = [f"Q{i}A" for i in ANX]
STR_C = [f"Q{i}A" for i in STR]

model_A = "\n".join([f"DEP =~ {' + '.join(DEP_C)}",
                     f"ANX =~ {' + '.join(ANX_C)}",
                     f"STR =~ {' + '.join(STR_C)}"])
model_B = f"G =~ {' + '.join(ALL)}"
model_C = "\n".join([f"G =~ {' + '.join(ALL)}",
                     f"SDEP =~ {' + '.join(DEP_C)}",
                     f" SANX =~ {' + '.join(ANX_C)}".replace(" SANX","SANX"),
                     f"SSTR =~ {' + '.join(STR_C)}",
                     "G ~~ 0*SDEP","G ~~ 0*SANX","G ~~ 0*SSTR",
                     "SDEP ~~ 0*SANX","SDEP ~~ 0*SSTR","SANX ~~ 0*SSTR"])

def fit(desc, model):
    t0 = time.time()
    try:
        m = semopy.Model(model); m.fit(df)
    except Exception as e:
        print(f"\n=== {desc}: FIT FAILED — {type(e).__name__}: {e} ==="); return None, None
    try:
        row = semopy.calc_stats(m).iloc[0]
        print(f"\n=== {desc} ({time.time()-t0:.1f}s) ===")
        for k in ["DoF","chi2","CFI","TLI","RMSEA","GFI","AGFI"]:
            if k in row.index:
                v = row[k]
                print(f"  {k:12s} {float(v):.4f}" if isinstance(v,(int,float,np.floating)) else f"  {k:12s} {v}")
        return m, row
    except Exception as e:
        print(f"\n=== {desc}: STATS FAILED — {e} ==="); return m, None

mA, sA = fit("A: 3 correlated (Dep/Anx/Str)", model_A)
mB, sB = fit("B: unidimensional", model_B)
mC, sC = fit("C: bifactor (G + 3 specifics)", model_C)

print("\n=== Comparison ===")
rows = []
for name, s in [("A:3-correlated", sA), ("B:unidim", sB), ("C:bifactor", sC)]:
    if s is None: continue
    rows.append({"model": name,
                 "chi2": float(s.get("chi2", np.nan)),
                 "df":   float(s.get("DoF", np.nan)),
                 "CFI":  float(s.get("CFI", np.nan)),
                 "TLI":  float(s.get("TLI", np.nan)),
                 "RMSEA":float(s.get("RMSEA", np.nan))})
comp = pd.DataFrame(rows)
print(comp.round(4).to_string(index=False))
comp.to_csv(OUT / "dass_cfa_comparison.csv", index=False)

if mC is not None:
    try:
        loadings = mC.inspect(std_est=True)
        loadings.to_csv(OUT / "dass_bifactor_loadings.csv", index=False)
        print(f"\nBifactor loadings saved: {loadings.shape}")
    except Exception as e:
        print(f"\nLoadings save failed: {e}")
