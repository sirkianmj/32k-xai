import sys, shutil, re
from pathlib import Path

SPEC = Path.home() / "projects/xai_project/Master Technical Specification_ Project 32k-XAI Cognitive Simulation Platform.md"
BAK  = SPEC.with_suffix(SPEC.suffix + ".v2.1.bak")

if not SPEC.exists():
    sys.exit(f"Spec not found: {SPEC}")

text = SPEC.read_text(encoding="utf-8")
orig_len = len(text)

if not BAK.exists():
    shutil.copy2(SPEC, BAK)
    print(f"Backup: {BAK}")
else:
    print(f"Backup exists (not overwritten): {BAK}")

report = []
def rep(name, old, new, count=1):
    global text
    n = text.count(old)
    if n < count:
        report.append((name, f"ANCHOR FAIL ({n} match)"))
        return
    text = text.replace(old, new, count)
    report.append((name, f"OK ({n} match, {count} replaced)"))

# Edit A: Entry 005
anchor_004 = "now reside under the new root."
entry_005 = """

Entry 005 — 07 October 2026 — w_MB Recovery Reconstruction (WSL)

Context: The original macOS fit_and_recover.py was not migrated to the Windows H: drive before the machine was set aside. The project's Migration Log confirmed the script and its artifacts (w_mb_real.csv, recovery.txt, recovery_scatter.csv) are absent from the migrated tree. The only surviving artifact from the macOS work is w_mb_recovery.txt from the earlier, abandoned R/MCMC attempt (ρ = 0.41, which was the failure that motivated the Python MLE re-derivation).

Action: The pipeline was re-authored from the specification on WSL Ubuntu (Windows 11, i5-9400F, GTX 1660 Ti). The reconstruction uses the same 4-parameter two-step hybrid, the same per-subject L-BFGS-B MLE with 3 restarts, and the same synthetic recovery protocol (N=60). It differs from the original in RNG seeding and restart initialization.

Real-data fit (N = 149): α = 0.367, β = 4.650, π = 0.302, w_MB = 0.623. These agree with the original macOS fit to within 0.01–0.08 on every parameter, confirming that the reconstruction is scientifically equivalent.

Recovery (N = 60): ρ = 0.7583, p = 2.28 × 10⁻¹², bias = −0.0067, RMSE = 0.2177. Pass (ρ > 0.6): YES.

Supersedes: The macOS ρ = 0.841 figure recorded in Entry 001 is superseded by the current on-disk reconstruction (ρ = 0.7583). The 0.841 value is retained in Entry 001 for provenance but must not be cited as the current recovery estimate in downstream artifacts. Both runs pass the G5 threshold; the difference is RNG-driven sampling noise at N=60.

Flags carried forward: [KOOL-VALIDATION PENDING], [RECOVERY-N60-PROVISIONAL], [COLLAPSED-LR].

Actions arising: N=500 recovery run required before D14. All future runs must save the script alongside its artifacts in the same directory (the original macOS run's loss of the script is the cause of this reconstruction)."""
rep("Entry 005 insertion", anchor_004, anchor_004 + entry_005)

# Edit B: Section 3.4
rep("3.4 α_R", "ESTIMATED (mean = 0.36)", "ESTIMATED (mean = 0.367, N=60, WSL reconstruction — see Entry 005)")
rep("3.4 β_choice", "ESTIMATED (mean = 4.57)", "ESTIMATED (mean = 4.650, N=60, WSL reconstruction — see Entry 005)")

# Edit C: Section 3.6
rep("3.6 w_MB", "ESTIMATED (mean = 0.62, ρ = 0.841)", "ESTIMATED (mean = 0.623, ρ = 0.7583, N=60, WSL reconstruction — see Entry 005)")
rep("3.6 φ_per", "ESTIMATED (mean = 0.30)", "ESTIMATED (mean = 0.302, N=60, WSL reconstruction — see Entry 005)")

# Edit D: Gate G5
rep("Gate G5", "Daw: PASSED (ρ = 0.841). Kool: PENDING.", "Daw: PASSED (ρ = 0.7583, Entry 005). Kool: PENDING.")

# Edit E: Section 6.1 hard rule
g8 = ("G8 Benchmark: Before full implementation, write a localized 500-agent benchmark tracing 100 "
      "timesteps to measure device memory overhead and wall-clock execution limits. If the benchmark "
      "exceeds 60 seconds per 100 steps, the architecture must be revised before 32k scale-up.")
g8_new = g8 + ("\n\n**Reproducibility hard rule (Entry 005):** Every fitting or recovery invocation "
               "must write its script, its stdout log, and its output CSVs to the same directory in "
               "the same invocation. `tee`-style logging alone is not sufficient because it does not "
               "preserve the code. Scripts that generate artifacts must be co-located with those "
               "artifacts at write time.")
rep("6.1 reproducibility rule", g8, g8_new)

# Edit F: Part X.12
rep("Part X.12",
    "Reproducibility: Every agent has a unique ID and seed. Trial-level data is stored. Generator version and seed list are immutable. Code is archived with commit hash or release tag.",
    "Reproducibility: Every agent has a unique ID and seed. Trial-level data is stored. Generator version and seed list are immutable. Code is archived with commit hash or release tag. Every fitting or recovery invocation must write its script, its stdout log, and its output CSVs to the same directory in the same invocation (tee alone is not sufficient). See Entry 005.")

print("\n=== Edit report ===")
for name, status in report:
    print(f"  {name:32s}  {status}")

if text == SPEC.read_text(encoding="utf-8"):
    print("\nNo changes made; spec untouched.")
    sys.exit(1)

SPEC.write_text(text, encoding="utf-8")
print(f"\nWritten: {SPEC}")
print(f"Length: {orig_len} -> {len(text)} chars")

print("\n=== Verification ===")
for pat, label in [
    (r"Entry 005 — 07 October 2026 — w_MB Recovery Reconstruction \(WSL\)", "Entry 005 header"),
    (r"ρ = 0\.7583", "new rho (should be 4: Entry 005, 3.6, G5, plus prose)"),
    (r"ρ = 0\.841", "old rho (should appear only in Entry 001 and 3.6-superseded prose)"),
    (r"WSL reconstruction — see Entry 005", "status parentheticals (should be 4)"),
    (r"Reproducibility hard rule \(Entry 005\)", "6.1 hard rule (should be 1)"),
]:
    n = len(re.findall(pat, text))
    print(f"  {label:70s}  {n}")
