import pandas as pd
import numpy as np
from pathlib import Path
import time, pickle, glob
from collections import Counter

DATA = Path.home() / "projects/xai_project/Data_Xai32kProje/XAI_Risk_Impulsivity_Data"
OUT  = Path.home() / "projects/xai_project/work_ssrt"

# Quality thresholds (Verbruggen et al. 2019)
MIN_GO_CORRECT   = 30
MIN_STOP_TRIALS  = 15
RT_MIN           = 0.100
RT_MAX           = 1.500
P_LOW            = 0.20
P_HIGH           = 0.80

t0 = time.time()
files = sorted(glob.glob(str(DATA / "sub-*_task-stopsignal_events.tsv")))
print(f"Found {len(files)} stop-signal files")

rows = []
n_skipped = 0

for f in files:
    sid = Path(f).name.split("_")[0]
    try:
        df = pd.read_csv(f, sep="\t")
    except Exception:
        n_skipped += 1
        continue

    cols_needed = {"trial_type", "ReactionTime", "TrialOutcome", "StopSignalDelay"}
    if not cols_needed.issubset(df.columns):
        n_skipped += 1
        continue

    df = df[df["trial_type"].notna()].copy()
    go   = df[df["trial_type"] == "GO"].copy()
    stop = df[df["trial_type"] == "STOP"].copy()

    # --- GO trial classification ---
    # Correct responses (SuccessfulGo, valid RT)
    go_correct = go[
        (go["TrialOutcome"] == "SuccessfulGo") &
        (go["ReactionTime"] >= RT_MIN) &
        (go["ReactionTime"] <= RT_MAX)
    ]
    go_rt_correct = go_correct["ReactionTime"].to_numpy()

    # Go choice errors (UnuccessfulGo but a response was made)
    go_err = go[
        (go["TrialOutcome"] == "UnuccessfulGo") &
        (go["ReactionTime"] >= RT_MIN) &
        (go["ReactionTime"] <= RT_MAX)
    ]
    go_rt_err = go_err["ReactionTime"].to_numpy()

    # Go omissions: RT < RT_MIN or missing
    go_omit = go[(go["ReactionTime"] < RT_MIN) | go["ReactionTime"].isna()]
    n_go_omit = len(go_omit)

    # All responded GO RTs (Rec. 8: include choice errors and premature)
    go_rt_responded = np.concatenate([go_rt_correct, go_rt_err])
    n_go_responded = len(go_rt_responded)

    # With replacement: append max(RT) once per omission
    if n_go_responded > 0 and n_go_omit > 0:
        max_rt = go_rt_responded.max()
        go_rt_replaced = np.concatenate([go_rt_responded,
                                          np.full(n_go_omit, max_rt)])
    else:
        go_rt_replaced = go_rt_responded.copy()

    # --- STOP trial classification ---
    n_stop    = len(stop)
    n_failed  = int((stop["TrialOutcome"] == "UnsuccessfulStop").sum())
    n_succ    = int((stop["TrialOutcome"] == "SuccessfulStop").sum())
    ssd_all   = stop["StopSignalDelay"].to_numpy()
    ssd_pos   = ssd_all[ssd_all > 0]
    mean_ssd_all = float(np.mean(ssd_all)) if n_stop else np.nan
    mean_ssd_pos = float(np.mean(ssd_pos)) if len(ssd_pos) else np.nan
    p_respond = n_failed / n_stop if n_stop else np.nan

    # --- Race-model check (Rec. 7) ---
    failed_stop = stop[stop["TrialOutcome"] == "UnsuccessfulStop"]
    failed_stop_rt = failed_stop[
        (failed_stop["ReactionTime"] >= RT_MIN) &
        (failed_stop["ReactionTime"] <= RT_MAX)
    ]["ReactionTime"].to_numpy()
    mean_rt_failed = float(np.mean(failed_stop_rt)) if len(failed_stop_rt) else np.nan
    mean_rt_go     = float(np.mean(go_rt_responded)) if n_go_responded else np.nan
    race_violation = (not np.isnan(mean_rt_failed) and not np.isnan(mean_rt_go)
                      and mean_rt_failed > mean_rt_go)

    # --- Flags ---
    flags = []
    if n_go_responded < MIN_GO_CORRECT: flags.append("FEW_GO")
    if n_stop < MIN_STOP_TRIALS:        flags.append("FEW_STOP")
    if not np.isnan(p_respond):
        if p_respond < P_LOW:  flags.append("P_LOW")
        if p_respond > P_HIGH: flags.append("P_HIGH")
    if len(ssd_pos) == 0 and n_stop: flags.append("NO_POS_SSD")
    if race_violation: flags.append("RACE_VIOLATION")

    # --- SSRT estimation ---
    ssrt = ssrt_noreplace = ssrt_pos = np.nan

    if (n_go_responded >= MIN_GO_CORRECT and n_stop >= MIN_STOP_TRIALS
            and 0 < p_respond < 1):
        p_use = min(max(p_respond, 0.01), 0.99)

        # Canonical: with replacement (Rec. 8)
        n_r = int(round((1 - p_use) * len(go_rt_replaced)))
        n_r = max(1, min(n_r, len(go_rt_replaced)))
        nth_rt_r = np.sort(go_rt_replaced)[n_r - 1]
        ssrt = nth_rt_r - mean_ssd_all

        # Sensitivity: no replacement
        n_nr = int(round((1 - p_use) * n_go_responded))
        n_nr = max(1, min(n_nr, n_go_responded))
        nth_rt_nr = np.sort(go_rt_responded)[n_nr - 1]
        ssrt_noreplace = nth_rt_nr - mean_ssd_all

        # Sensitivity: positive SSD only
        if not np.isnan(mean_ssd_pos):
            ssrt_pos = nth_rt_r - mean_ssd_pos

    rows.append({
        "subject":             sid,
        "n_go_responded":      n_go_responded,
        "n_go_omissions":      n_go_omit,
        "n_go_choice_err":     len(go_rt_err),
        "n_stop":              n_stop,
        "n_success_stop":      n_succ,
        "n_failed_stop":       n_failed,
        "p_respond":           p_respond,
        "mean_ssd_all":        mean_ssd_all,
        "mean_ssd_pos":        mean_ssd_pos,
        "go_rt_mean":          mean_rt_go,
        "go_rt_sd":            float(np.std(go_rt_responded, ddof=1)) if n_go_responded > 1 else np.nan,
        "mean_rt_failed_stop": mean_rt_failed,
        "race_violation":      bool(race_violation),
        "ssrt":                ssrt,
        "ssrt_noreplace":      ssrt_noreplace,
        "ssrt_pos":            ssrt_pos,
        "flags":               ",".join(flags) if flags else "",
    })

out = pd.DataFrame(rows)
print(f"Processed: {len(out)}   Skipped: {n_skipped}")
print(f"Elapsed:   {time.time()-t0:.1f}s")
print()

print("=== Per-subject distributions ===")
print(out[["n_go_responded","n_go_omissions","n_go_choice_err","n_stop",
           "p_respond","mean_ssd_all","go_rt_mean","ssrt"]].describe().round(4).to_string())
print()

print("=== SSRT quality ===")
print(f"SSRT computable:  {out['ssrt'].notna().sum()} / {len(out)}")
print(f"Flagged (any):    {(out['flags'] != '').sum()} / {len(out)}")
print(f"Race violation:   {out['race_violation'].sum()} / {len(out)}")
print()
print("=== Flag breakdown ===")
all_flags = Counter()
for s in out["flags"]:
    if s:
        for fl in s.split(","):
            all_flags[fl] += 1
for k, v in all_flags.most_common():
    print(f"  {k:16s}  {v}")
print()

print("=== SSRT canonical (integration, with replacement) ===")
ssrt_valid = out["ssrt"].dropna()
print(ssrt_valid.describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).round(4).to_string())
print()
print("=== SSRT sensitivity: no replacement ===")
ssrt_nr = out["ssrt_noreplace"].dropna()
print(ssrt_nr.describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).round(4).to_string())
print()
print("=== SSRT sensitivity: positive SSD only ===")
ssrt_pos_v = out["ssrt_pos"].dropna()
print(ssrt_pos_v.describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).round(4).to_string())
print()
print(f"corr(ssrt, ssrt_noreplace) = {ssrt_valid.corr(out.loc[ssrt_valid.index, 'ssrt_noreplace']):.4f}")
print(f"corr(ssrt, ssrt_pos)       = {ssrt_valid.corr(out.loc[ssrt_valid.index, 'ssrt_pos']):.4f}")
print()
print(f"ssrt <= 0:           {(ssrt_valid <= 0).sum()}")
print(f"ssrt_noreplace <= 0: {(ssrt_nr <= 0).sum()}")
print(f"ssrt_pos <= 0:       {(ssrt_pos_v <= 0).sum()}")

out.to_csv(OUT / "ssrt_subjects.csv", index=False)
with open(OUT / "ssrt_subjects.pkl", "wb") as fh:
    pickle.dump(out, fh, protocol=4)
print()
print(f"Saved: {len(out)} rows -> ssrt_subjects.csv + .pkl")
