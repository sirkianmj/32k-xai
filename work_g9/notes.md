# G9 — Data Provenance Matrix — notes

**Deliverable:** A single table mapping every parameter in the dictionary to
its data source, license / DUA status, and provenance notes. Output files:
`provenance_matrix.csv`, `provenance_matrix.md`.

## Audit summary

| Category | Count |
| :--- | ---: |
| Total parameters in dictionary | 31 |
| Extracted with empirical evidence | 20 |
| Documented synthetic priors | 10 |
| Documented not-identifiable | 1 |
| **Unaccounted for** | **0** |

**G9 criterion:** no parameter without a documented source.
**Result: PASS.**

## Evidence linkage

For every extracted parameter, `notes.md` exists in the corresponding
`work_*/` directory and contains source, filters, methods, and reproducibility
instructions:

| work directory | parameters documented |
| :--- | :--- |
| work_ipip | 5 (OCEAN) |
| work_sd3 | 4 (D Dark Core, S_M, S_P, S_N) |
| work_dass | 3 (DASS specifics) |
| work_g4 | 1 (general distress) |
| work_n500 | 4 (w_MB, α_R, β_choice, φ_per) |
| work_ssrt | 1 (a_inhib) |
| work_dd | 1 (k_DD) |
| work_fs | 1 (λ_ineq) |

## License breakdown

| License | Count |
| :--- | ---: |
| Open Psychometrics public | 13 |
| No source (synthetic priors) | 10 |
| Academic | 3 |
| HCP restricted DUA | 2 |
| Academic (Zorowitz & Niv 2021) | 1 |
| Unknown (Qualtrics export) | 1 |
| Not identifiable | 1 |

**Known licensing caveats to flag for downstream use:**

1. **HCP-derived parameters** (`a_inhib` from Stop-Signal, `λ_ineq` from UGR):
   governed by HCP restricted Data Use Agreement. Redistribution of raw data
   is prohibited; the derived parameter estimates in this repo are released
   under MIT, but the source data is not.
2. **`k_DD` (delay discounting)**: the raw file is a Qualtrics survey export
   with no license statement in the directory. Flagged
   `[LICENSE-UNKNOWN]`. Should be resolved before any publication.
3. **Open Psychometrics samples** (OCEAN, DASS, SD3): public, research-
   permissible, no DUA constraints.
4. **Zorowitz & Niv 2021 two-step dataset**: academic use only, cited in
   the notes.

## Parameters lacking empirical coverage

Ten parameters have no extractable empirical source in the current asset
inventory and are designated `[SYNTHETIC PRIOR]`:

θ_T, u_U, η_vol, ω_social, λ_norm, s_spite, τ_sens, γ_sal, κ_ctx, m_t

These will be sampled in the Vine Copula from literature-derived marginal
distributions and couplings. When the copula is constructed, each of these
must carry a specific literature citation (not just a generic flag) at that
point.

## Parameters documented as not identifiable

**β_FS** (Fehr-Schmidt guilt) — the HCP UGR design contains no advantageous-
inequity trials (all offers ≤ half-endowment), so β is not identified.
Flag: `[NOT-IDENTIFIABLE-IN-HCP-UGR]`. Fallback: single-parameter λ_ineq.

## Interpreting this matrix

- **Row for every parameter** — no parameter can enter the Vine Copula or
  any downstream artefact without a row here.
- **No row is a bare estimate** — every row carries license, source, and
  at least one provenance note or flag.
- **Synthetic priors are legitimate but marked** — the spec permits them
  but requires them to be documented as such in every XAI trace.

## Reproducibility

Script: `provenance_matrix.py`. Log: `provenance_matrix.log`. Runtime: <1 s.
cd work_g9 && python provenance_matrix.py


The script reads the file system (`work_*/notes.md` existence) and a
hand-curated list of parameters. It fails (via the "Unaccounted" count)
if any parameter is added to the list without a corresponding work_dir
and notes.md, or without an explicit `NO SOURCE` / `NOT IDENTIFIABLE`
designation.
