import pandas as pd
from pathlib import Path

ROOT = Path.home() / "projects/xai_project"
DATA = ROOT / "Data_Xai32kProje"

rows = [
    ("O","Openness","work_ipip","IPIP-FFM-data-8Nov2018","Open Psychometrics public",
     "score_ipip.py; 603,322 IPC=1 subjects"),
    ("C","Conscientiousness","work_ipip","IPIP-FFM-data-8Nov2018","Open Psychometrics public",
     "score_ipip.py; 50-item Big Five"),
    ("E","Extraversion","work_ipip","IPIP-FFM-data-8Nov2018","Open Psychometrics public",
     "score_ipip.py"),
    ("A","Agreeableness","work_ipip","IPIP-FFM-data-8Nov2018","Open Psychometrics public",
     "score_ipip.py"),
    ("N","Neuroticism","work_ipip","IPIP-FFM-data-8Nov2018","Open Psychometrics public",
     "score_ipip.py; anti-double-count with DASS subscale applies"),
    ("D","General Dark Core","work_sd3","SD3","Open Psychometrics public",
     "score_sd3.py + work_g3/cfa_sd3.py (bifactor)"),
    ("S_M","Machiavellianism","work_sd3","SD3","Open Psychometrics public",
     "score_sd3.py; alpha=0.861"),
    ("S_P","Psychopathy","work_sd3","SD3","Open Psychometrics public",
     "score_sd3.py; alpha=0.798"),
    ("S_N","Narcissism","work_sd3","SD3","Open Psychometrics public",
     "score_sd3.py; alpha=0.805"),
    ("general_distress","DASS general factor","work_g4","XAI_Mental_Health_DASS",
     "Open Psychometrics public","cfa_dass.py (bifactor); 39,775 subjects"),
    ("DASS_Dep","DASS depression specific","work_dass","XAI_Mental_Health_DASS",
     "Open Psychometrics public","score_dass.py; secondary modulator"),
    ("DASS_Anx","DASS anxiety specific","work_dass","XAI_Mental_Health_DASS",
     "Open Psychometrics public","score_dass.py; secondary modulator"),
    ("DASS_Str","DASS stress specific","work_dass","XAI_Mental_Health_DASS",
     "Open Psychometrics public","score_dass.py; secondary modulator"),
    ("w_MB","Model-based arbitration","work_n500","XAI_Markov_TwoStep_Data",
     "Academic (Zorowitz & Niv 2021)","fit_and_recover.py; rho=0.7824 N=500"),
    ("alpha_R","Learning rate (collapsed)","work_n500","XAI_Markov_TwoStep_Data",
     "Academic","fit_and_recover.py; [COLLAPSED-LR]"),
    ("beta_choice","Inverse temperature","work_n500","XAI_Markov_TwoStep_Data",
     "Academic","fit_and_recover.py; [COLLAPSED-LR]"),
    ("phi_per","Perseveration","work_n500","XAI_Markov_TwoStep_Data",
     "Academic","fit_and_recover.py"),
    ("k_DD","Delay discounting","work_dd","XAI_Delay_Discounting_Data",
     "Unknown - Qualtrics export, no license stated","score_dd.py; [ADAPTIVE-DESIGN] [NO-REFIT]"),
    ("a_inhib","SSRT (inhibitory control)","work_ssrt","XAI_Risk_Impulsivity_Data",
     "HCP restricted DUA","score_ssrt.py; integration method per Verbruggen 2019"),
    ("lambda_ineq","Inequity aversion (alpha only)","work_fs","XAI_Social_Games_Data",
     "HCP restricted DUA","score_fs.py; [SIMPLIFIED MODEL] [ENVY-ONLY]"),
    ("theta_T","Threat sensitivity",None,None,"NO SOURCE","[SYNTHETIC PRIOR] literature-derived"),
    ("u_U","Uncertainty aversion",None,None,"NO SOURCE","[SYNTHETIC PRIOR] literature-derived"),
    ("eta_vol","Volatility update",None,None,"NO SOURCE","[SYNTHETIC PRIOR] literature-derived"),
    ("omega_social","Social value weight",None,None,"NO SOURCE","[SYNTHETIC PRIOR] literature-derived"),
    ("lambda_norm","Moral norm sensitivity",None,None,"NO SOURCE","[SYNTHETIC PRIOR] literature-derived"),
    ("s_spite","Spite sensitivity",None,None,"NO SOURCE","[SYNTHETIC PRIOR] literature-derived"),
    ("tau_sens","Sensory precision",None,None,"NO SOURCE","[SYNTHETIC PRIOR] literature-derived"),
    ("gamma_sal","Salience gain",None,None,"NO SOURCE","[SYNTHETIC PRIOR] literature-derived"),
    ("kappa_ctx","Context integration",None,None,"NO SOURCE","[SYNTHETIC PRIOR] literature-derived"),
    ("m_t","Mood/state",None,None,"NO SOURCE","[SYNTHETIC PRIOR] AR(1) state model"),
    ("beta_FS","Fehr-Schmidt guilt",None,"XAI_Social_Games_Data","NOT IDENTIFIABLE",
     "[NOT-IDENTIFIABLE-IN-HCP-UGR] no advantageous-inequity offers"),
]

df = pd.DataFrame(rows, columns=[
    "Parameter","Construct","Work dir","Raw dataset",
    "License / DUA status","Provenance notes"
])

print("=== G9 provenance audit ===")
print(f"Total parameters in dictionary: {len(df)}")

def has_evidence(row):
    wd = row["Work dir"]
    if pd.isna(wd):
        return False
    d = ROOT / wd
    return d.exists() and (d / "notes.md").exists()

df["Has evidence"] = df.apply(has_evidence, axis=1)
extracted = df[df["Has evidence"]]
print(f"With evidence (work_*/notes.md exists): {len(extracted)}")

print("\nParameters per work dir:")
for k, v in sorted(df[df["Work dir"].notna()].groupby("Work dir").size().to_dict().items()):
    print(f"  {k:12s} {v}")

print("\n=== Data directories on disk ===")
if DATA.exists():
    for d in sorted(DATA.iterdir()):
        if d.is_dir():
            print(f"  {d.name}")

print("\n=== License status ===")
for k, v in df["License / DUA status"].value_counts().items():
    print(f"  {k:45s} {v}")

df.to_csv("provenance_matrix.csv", index=False)
print("\nSaved provenance_matrix.csv")

n_extracted = len(extracted)
n_synthetic = (df["License / DUA status"] == "NO SOURCE").sum()
n_not_id = (df["License / DUA status"] == "NOT IDENTIFIABLE").sum()
n_unaccounted = len(df) - n_extracted - n_synthetic - n_not_id

with open("provenance_matrix.md", "w") as f:
    f.write("# G9 - Data Provenance Matrix\n\n")
    f.write(f"**Total parameters:** {len(df)}\n\n")
    f.write(f"**With empirical evidence:** {n_extracted}\n\n")
    f.write(f"**Synthetic priors (no dataset):** {n_synthetic}\n\n")
    f.write(f"**Not identifiable:** {n_not_id}\n\n")
    f.write(f"**Unaccounted:** {n_unaccounted}\n\n")
    f.write("| Parameter | Construct | Work dir | Raw dataset | License / DUA | Notes |\n")
    f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for _, r in df.iterrows():
        wd = r["Work dir"] if not pd.isna(r["Work dir"]) else "-"
        rd = r["Raw dataset"] if not pd.isna(r["Raw dataset"]) else "-"
        f.write(f"| {r['Parameter']} | {r['Construct']} | {wd} | {rd} | {r['License / DUA status']} | {r['Provenance notes']} |\n")
print("Saved provenance_matrix.md")

print("\n=== G9 verdict ===")
print(f"  Extracted with evidence:        {n_extracted}")
print(f"  Synthetic priors (documented):  {n_synthetic}")
print(f"  Not identifiable (documented):  {n_not_id}")
print(f"  Parameters without any status:  {n_unaccounted}")
print("  G9 criterion: no parameter without a documented source.")
print(f"  Result: {'PASS' if n_unaccounted == 0 else 'FAIL'}")
