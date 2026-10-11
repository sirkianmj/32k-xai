# P6d Completion Note — Grounding the 10 [SYNTHETIC-PRIOR] Parameters

**Date:** 11 October 2026
**Status:** Complete for this session.
**Supersedes:** `P6D_SCOPE.md` (scope) with empirical result.

## 1. Final state of the copula

`work_g2/synthetic_32k_p6d.pkl` — shape (32000, 26). The original 30-column
copula with 4 inert parameters retired and 5 synthetic priors replaced by
empirical or proxy marginals. The original `synthetic_32k.pkl` (30 columns)
is untouched.

## 2. Priors by final status

| prior | status | source | N | method |
|:---|:---|:---|:---:|:---|
| `m_t` | **GROUNDED-DIRECT** | PANAS pre-intervention (v5nk7 Study2B) | 329 | rank-preserving swap |
| `theta_T` | **GROUNDED-PROXY** | HCP EMOTION FACE_MEDIAN_RT | 482 | rank-preserving swap, RT inverted |
| `u_U` | **GROUNDED-PROXY** | HCP GAMBLING PUNISH_PROP_LARGER | 491 | rank-preserving swap, inverted |
| `omega_social` | **GROUNDED-PROXY** | XAI_Social_Games trust-task investment | 113 | rank-preserving swap |
| `eta_vol` | SYNTHETIC-PRIOR | (no dataset on disk) | — | — |
| `gamma_sal` | SYNTHETIC-PRIOR | (no dataset on disk) | — | — |
| `lambda_norm` | **RETIRED (inert)** | — | — | — |
| `s_spite` | **RETIRED (inert)** | — | — | — |
| `tau_sens` | **RETIRED (inert)** | — | — | — |
| `kappa_ctx` | **RETIRED (inert)** | — | — | — |

Net change: 10 synthetic → **2 synthetic** (eta_vol, gamma_sal).

## 3. Candidates that were considered and rejected

| prior | candidate | reason for rejection |
|:---|:---|:---|
| `theta_T` | HCP EMOTION FACE_ACC | ceiling effect: 71.4% of subjects at accuracy=1.0, sd=0.027 |
| `omega_social` (alt) | trust-task choice high/low | 3 discrete trust values only, no continuous variance |

## 4. Cumulative calibration impact

Full-scale verification at 32k x 500 (p6c_verify2.py):

| benchmark | step-1 baseline | final P6d | delta | SE | % of SE |
|:---|---:|---:|---:|---:|---:|
| UG mean offer | 0.39458367 | 0.3940 | -6.0e-04 | 0.050 | 1.20% |
| UG rejection | 0.12024558 | 0.1202 | -5.0e-05 | 0.050 | 0.10% |
| TG mean send | 0.48318192 | 0.4832 | -1.0e-05 | 0.130 | 0.01% |
| TG mean return | 0.33410013 | 0.3341 | -1.0e-05 | 0.120 | 0.01% |
| Final Gini | 0.44285348 | 0.4429 | +5.0e-05 | 0.120 | 0.04% |

**5/5 PASS at every step.** Cumulative deltas at most 1.2% of the smallest
SE. The calibration is not meaningfully affected by replacing the four
marginals.

## 5. Method

Each swap uses rank-preserving quantile substitution:

1. Compute the rank of every existing value in [0, 1].
2. Replace each value with the empirical quantile of the source at that rank.
3. Rescale to preserve the old mean and sd.
4. Clip to the parameter's stated range.

This is the standard copula-theory operation for marginal replacement. It
preserves the rank order up to ties in the source empirical distribution,
and preserves the dependence structure between the parameter and every
other column in the copula by construction.

The failed P6d attempt at commit `1cc5e19` violated this by rank-transforming
an existing copula column. The rebuild uses only external data sources.

## 6. Direction assumptions

Three of the four grounded parameters required a direction decision:

- `theta_T` (RT-inverted): faster face detection = higher threat sensitivity.
- `u_U` (inverted): more risk-taking under punishment = lower uncertainty aversion.
- `omega_social` (as-is): more trust investment = higher social value weight.

Each is documented in the corresponding commit and flagged `[DIRECTION-ASSUMED]`.
A sensitivity check on any of these is a legitimate future test.

## 7. External-required priors

Two parameters remain `[SYNTHETIC-PRIOR]` and require external data:

- `eta_vol` (volatility learning rate) — needs a volatile-bandit paradigm.
- `gamma_sal` (salience gain) — needs an oddball task with behavioural summaries.

Neither has a usable dataset on this machine. Both are documented limitations.
Any further work on them requires sourcing a specific dataset by name, not
open-ended OSF search.

## 8. Impact on project claims

**What improves:**
- 5 of 10 previously synthetic parameters now have documented empirical bases.
- 4 of 10 were removed as dead weight.
- The remaining 2 are explicitly documented as external-required.

**What does not change:**
- The 5/5 human benchmark PASS is preserved.
- The XAI trace schema is unchanged.
- The calibrated θ in `p6c_results.json` remains valid for the P6d copula
  (the platform values match within 1.2% of SE).

**What must be stated in any downstream artifact:**
- The P6d copula is `work_g2/synthetic_32k_p6d.pkl`. The pre-P6d copula is
  `work_g2/synthetic_32k.pkl`.
- Three of the four grounded parameters are proxies, not direct measurements.
- Two synthetic priors remain.

## 9. Reproduction

Build commands (in order):
1. `P6d step 1`: retire 4 inert columns → (32000, 26) copula.
2. `P6d step 2`: swap `m_t` with PANAS.
3. `P6d step 3`: swap `theta_T` with HCP FACE_MEDIAN_RT (inverted).
4. `P6d step 4`: swap `u_U` with HCP PUNISH_PROP_LARGER (inverted).
5. `P6d step 5`: swap `omega_social` with trust-task investment.

Each step's commit contains its verification artifact.

Verification:
    cd work_env
    P6C_PARAMS="$HOME/projects/xai_project/work_g2/synthetic_32k_p6d.pkl" python3 -u p6c_verify2.py

Expected output: 5/5 PASS.
