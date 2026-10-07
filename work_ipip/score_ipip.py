import pandas as pd
import numpy as np
from pathlib import Path
import time, pickle

SRC = Path.home() / "projects/xai_project/Data_Xai32kProje/IPIP-FFM-data-8Nov2018/data-final.csv"
OUT = Path.home() / "projects/xai_project/work_ipip"

ITEMS = (
    [f"EXT{i}" for i in range(1, 11)] +
    [f"EST{i}" for i in range(1, 11)] +
    [f"AGR{i}" for i in range(1, 11)] +
    [f"CSN{i}" for i in range(1, 11)] +
    [f"OPN{i}" for i in range(1, 11)]
)
REV = {"EXT2","EXT4","EXT6","EXT8","EXT10",
       "EST2","EST4",
       "AGR1","AGR3","AGR5","AGR7",
       "CSN2","CSN4","CSN6","CSN8",
       "OPN2","OPN4","OPN6"}
TRAITS = {"O":[f"OPN{i}" for i in range(1,11)],
          "C":[f"CSN{i}" for i in range(1,11)],
          "E":[f"EXT{i}" for i in range(1,11)],
          "A":[f"AGR{i}" for i in range(1,11)],
          "N":[f"EST{i}" for i in range(1,11)]}

t0 = time.time()
parts_all, parts_ipc1 = [], []
n_in = n_nan = n_range = 0

for chunk in pd.read_csv(SRC, sep="\t", usecols=ITEMS+["IPC"],
                         chunksize=200_000, low_memory=False):
    n_in += len(chunk)
    chunk = chunk.dropna(subset=ITEMS)
    n_nan += len(chunk)
    mask = np.ones(len(chunk), dtype=bool)
    for c in ITEMS:
        v = chunk[c].to_numpy()
        mask &= (v >= 1) & (v <= 5) & (v == np.floor(v))
    chunk = chunk[mask]
    n_range += len(chunk)
    for c in REV:
        chunk[c] = 6 - chunk[c]
    for trait, cols in TRAITS.items():
        chunk[trait] = chunk[cols].sum(axis=1)
    sub = chunk[list(TRAITS.keys())+["IPC"]]
    parts_all.append(sub[list(TRAITS.keys())])
    parts_ipc1.append(sub[sub["IPC"]==1][list(TRAITS.keys())])

all_df  = pd.concat(parts_all, ignore_index=True)
ipc1_df = pd.concat(parts_ipc1, ignore_index=True)
for t in TRAITS:
    all_df[t]  = (all_df[t]  - 10) / 40
    ipc1_df[t] = (ipc1_df[t] - 10) / 40

print(f"Rows read:                  {n_in}")
print(f"After NaN drop:             {n_nan}")
print(f"After [1,5]-integer filter: {n_range}")
print(f"After IPC==1 filter:        {len(ipc1_df)}")
print(f"Elapsed:                    {time.time()-t0:.1f}s")
print()
print("=== OCEAN, IPC==1 ===")
print(ipc1_df.describe().round(4).to_string())
print()
print("=== correlation matrix (IPC==1) ===")
print(ipc1_df.corr().round(3).to_string())
print()
assert ipc1_df.min().min() >= 0.0 and ipc1_df.max().max() <= 1.0, "Range violation!"
print("Range check: all values in [0,1]  OK")

ipc1_df.to_csv(OUT/"ipip_ocean_ipc1.csv", index=False)
with open(OUT/"ipip_ocean_ipc1.pkl", "wb") as fh:
    pickle.dump(ipc1_df, fh, protocol=4)
ipc1_df.sample(n=min(50_000, len(ipc1_df)), random_state=42)\
      .to_csv(OUT/"ipip_ocean_ipc1_sample50k.csv", index=False)
print(f"Saved: ipip_ocean_ipc1.csv + .pkl + 50k sample")
