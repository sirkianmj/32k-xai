import pandas as pd
import numpy as np
from pathlib import Path
import time, pickle, glob

DATA = Path.home() / "projects/xai_project/Data_Xai32kProje/XAI_Risk_Impulsivity_Data"
OUT  = Path.home() / "projects/xai_project/work_ssrt"

# Quality thresholds (Verbruggen et al. 2019, eLife)
MIN_GO_CORRECT   = 30     # minimum valid GO RTs
MIN_STOP_TRIALS  = 15     # minimum STOP trials
RT_MIN           = 0.100  # exclude anticipatory (<100 ms)
RT_MAX           = 1.500  # exclude excessively slow (>1500 ms)
P_LOW            = 0.20   # p(respond|signal) below this: staircase too easy
P_HIGH           = 0.80   # above this: staircase too hard

t0 = time.time()
files = sorted(glob.glob(str(DATA / "sub-*_task-stopsignal_events.tsv")))
print(f"Found {len(files)} stop-signal files")

rows = []
n_skipped = 0

for f in files:
    sid = Path(f).name.split("_")[0]  # e.g. sub-10159
    try:
        df = pd.read_csv(f, sep="\t")
    except Exception as e:
        n_skipped += 1
        continue

    # Normalize potential column-name variants
    cols_needed = {"trial_type", "ReactionTime", "TrialOutcome", "StopSignalDelay"}
    if not cols_needed.issubset(df.columns):
        n_skipped += 1
        continue

    # Clean trial_type: drop NaN/blank rows
    df = df[df["trial_type"].notna()].copy()

    go   = df[df["trial_type"] == "GO"]
    stop = df[df["trial_type"] == "STOP"]

    # GO RTs: correct GO, in [RT_MIN, RT_MAX]
    go_ok = go[
        (go["TrialOutcome"] == "SuccessfulGo") &
        (go["ReactionTime"] >= RT_MIN) &
        (go["ReactionTime"] <= RT_MAX)
    ]
    go_rt = go_ok["ReactionTime"].to_numpy()
    n_go_correct = len(go_rt)

    # STOP trials: use all with valid SSD
    n_stop = len(stop)
    n_failed = (stop["TrialOutcome"] == "UnsuccessfulStop").sum()
    n_succ   = (stop["TrialOutcome"] == "SuccessfulStop").sum()

    ssd_all     = stop["StopSignalDelay"].to_numpy()
    ssd_positive = ssd_all[ssd_all > 0]

    mean_ssd_all = float(np.mean(ssd_all)) if n_stop else np.nan
    mean_ssd_pos = float(np.mean(ssd_positive)) if len(ssd_positive) else np.nan

    p_respond = n_failed / n_stop if n_stop else np.nan

    # Flag conditions
    flags = []
    if n_go_correct < MIN_GO_CORRECT: flags.append("FEW_GO")
    if n_stop < MIN_STOP_TRIALS:      flags.append("FEW_STOP")
    if not np.isnan(p_respond):
        if p_respond < P_LOW:  flags.append("P_LOW")
        if p_respond > P_HIGH: flags.append("P_HIGH")
    if len(ssd_positive) == 0 and n_stop: flags.append("NO_POS_SSD")

    # SSRT via integration method
    ssrt = np.nan
    ssrt_pos = np.nan
    if n_go_correct >= MIN_GO_CORRECT and n_stop >= MIN_STOP_TRIALS and 0 < p_respond < 1:
        # Clip p to avoid 0/1 endpoints
        p_use = min(max(p_respond, 0.01), 0.99)
        n_target = int(round((1 - p_use) * n_go_correct))
        n_target = max(1, min(n_target, n_go_correct))
        go_sorted = np.sort(go_rt)
        nth_rt = go_sorted[n_target - 1]

        ssrt      = nth_rt - mean_ssd_all
        if not np.isnan(mean_ssd_pos):
            ssrt_pos = nth_rt - mean_ssd_pos

    rows.append({
        "subject":       sid,
        "n_go_correct":  n_go_correct,
        "n_stop":        n_stop,
        "n_success_stop": int(n_succ),
        "n_failed_stop":  int(n_failed),
        "p_respond":     p_respond,
        "mean_ssd_all":  mean_ssd_all,
        "mean_ssd_pos":  mean_ssd_pos,
        "go_rt_mean":    float(np.mean(go_rt)) if n_go_correct else np.nan,
        "go_rt_sd":      float(np.std(go_rt, ddof=1)) if n_go_correct > 1 else np.nan,
        "ssrt":          ssrt,
        "ssrt_pos":      ssrt_pos,
        "flags":         ",".join(flags) if flags else "",
    })

out = pd.DataFrame(rows)
print(f"Processed: {len(out)}   Skipped: {n_skipped}")
print(f"Elapsed:   {time.time()-t0:.1f}s")
print()

print("=== Per-subject distributions ===")
print(out[["n_go_correct","n_stop","p_respond","mean_ssd_all","go_rt_mean","ssrt"]].describe().round(4).to_string())
print()

print("=== SSRT quality ===")
n_ssrt_valid = out["ssrt"].notna().sum()
print(f"SSRT computable:  {n_ssrt_valid} / {len(out)}")
n_flagged = (out["flags"] != "").sum()
print(f"Flagged subjects: {n_flagged} / {len(out)}")
print()
print("=== Flag breakdown ===")
from collections import Counter
all_flags = Counter()
for s in out["flags"]:
    if s:
        for f in s.split(","):
            all_flags[f] += 1
for k, v in all_flags.most_common():
    print(f"  {k:12s}  {v}")
print()

print("=== SSRT distribution (valid only) ===")
ssrt_valid = out["ssrt"].dropna()
print(ssrt_valid.describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).round(4).to_string())
print()
print(f"SSRT <= 0 (implausible):  {(ssrt_valid <= 0).sum()}")
print(f"SSRT >  1.0s (implausible): {(ssrt_valid > 1.0).sum()}")
print()
print("=== SSRT with positive SSD only (sensitivity) ===")
ssrt_pos_valid = out["ssrt_pos"].dropna()
print(ssrt_pos_valid.describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).round(4).to_string())
print()
print(f"corr(ssrt, ssrt_pos) = {ssrt_valid.corr(out.loc[ssrt_valid.index, 'ssrt_pos']):.4f}")

# Robust save
out.to_csv(OUT / "ssrt_subjects.csv", index=False)
with open(OUT / "ssrt_subjects.pkl", "wb") as fh:
    pickle.dump(out, fh, protocol=4)
print()
print(f"Saved: ssrt_subjects.csv + .pkl ({len(out)} rows)")
