# P6D Rollback Note — 11 Oct 2026

## What was attempted

An uncommitted script `work_g2/ground_priors.py` attempted to "ground"
the ten `[SYNTHETIC-PRIOR]` parameters by replacing each column in
`work_g2/synthetic_32k.pkl` with a monotone rank-transform of an existing
column already present in the same copula:

| Prior | Source column used |
|---|---|
| theta_T | DASS_Anx |
| u_U | 0.6·DASS_Anx + 0.4·inv(O) by rank |
| eta_vol | C |
| omega_social | A |
| lambda_norm | rank(C + A) |
| s_spite | rank(D + S_P − A) |
| tau_sens | O |
| gamma_sal | E |
| kappa_ctx | O |
| m_t | DASS_Dep centered |

No new empirical dataset was ingested. The script overwrote
`synthetic_32k.pkl` in place (backup written as `.pre_p6d`).

## Why it was rejected

The technique collapses the effective dimensionality of the population:
each "prior" becomes a deterministic function of a column already in the
copula. The Spec (Part III) requires these parameters to be *coupled to*
psychometric variables through the vine copula, not *equal to* them by
rank transform.

The change breaks the calibration. Same theta as in `p6c_results.json`,
same 32k x 500 verifier, only the copula differs:

| Benchmark | pre-P6d | P6d-modified |
|---|---|---|
| UG mean offer | 0.3946 PASS | 0.4036 PASS |
| UG rejection  | 0.1202 PASS | 0.1393 PASS |
| TG mean send  | 0.4832 PASS | 0.4903 PASS |
| TG mean return| 0.3341 PASS | 0.3561 PASS |
| **Final Gini**| **0.4429 PASS** | **0.6292 FAIL (2 SE off)** |

The Gini benchmark moves 0.186 in the wrong direction purely from the
copula substitution. Unique-value counts for the ten columns collapse
from ~15,700 to ~40, confirming the re-parameterization.

## Action taken

- Restored `work_g2/synthetic_32k.pkl` from the on-disk backup `.pre_p6d`
  (byte-identical to the version committed at `7a730f8`).
- Re-ran `work_env/p6c_verify2.py` at 32k x 500. Result: 5/5 PASS,
  byte-identical to the `p6c_final.json` committed at `7a730f8`.
- Preserved both the failed attempt (`.p6d_attempt`, 5.6 MB) and the
  correct pre-P6d version (`.pre_p6d`, 6.9 MB) as untracked binaries
  for forensics.
- Preserved `ground_priors.py` and `ground_priors.log` in this commit
  as evidence of the rejected approach.

## Seed robustness of the restored calibration

32k x 500, theta from `p6c_results.json`:

| seed | UG offer | UG rej | TG send | TG ret | Gini | pass |
|---|---|---|---|---|---|---|
| 42 | 0.3946 | 0.1202 | 0.4832 | 0.3341 | 0.4429 | 5/5 |
| 43 | 0.3935 | 0.1211 | 0.4826 | 0.3350 | 0.4433 | 5/5 |
| 44 | 0.3952 | 0.1205 | 0.4842 | 0.3329 | 0.4439 | 5/5 |
| 45 | 0.3938 | 0.1219 | 0.4836 | 0.3334 | 0.4433 | 5/5 |
| 46 | 0.3941 | 0.1199 | 0.4839 | 0.3364 | 0.4410 | 5/5 |

Per-seed SD <= 0.0013 on every benchmark. The 5/5 PASS is not a
seed-42 artifact.

## What this does NOT do

This note does not close the P6d question. The ten `[SYNTHETIC-PRIOR]`
parameters remain ungrounded. A correct P6d must ingest external,
independent empirical data for each construct and refit the marginal.
Rank-transforming existing columns is not grounding and is not permitted
under the spec's empirical-grounding principle.

Flag: `[P6D-RANK-TRANSFORM-REJECTED]`.
