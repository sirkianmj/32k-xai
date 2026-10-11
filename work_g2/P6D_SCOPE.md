# P6d Scope — Grounding the 10 [SYNTHETIC-PRIOR] Parameters

**Date:** 11 October 2026
**Status:** Scope document. No copula change. No code change to the engine.
**Supersedes:** earlier draft of this file (uncommitted).
**Predecessors:** P6d attempt rejected (commit `1cc5e19`); greps and archive
inspections logged in `p6d_scope*.log`, `p6d_archives_*.log`, `p6d_mt_*.log`.

## 1. Executive summary

Of the 10 parameters flagged `[SYNTHETIC-PRIOR]` in `work_g9/provenance_matrix.md`:

- **4 are inert in the calibrated engine** (`lambda_norm`, `s_spite`, `tau_sens`,
  `kappa_ctx`). They exist as columns in the copula but are never read by
  `p6c_v5.step`. Grounding them changes nothing measurable.
- **1 is now groundable directly** (`m_t`, PANAS, N=329). Quantiles match the
  current copula within ±0.13 at every tested percentile.
- **3 are groundable as proxies** (`theta_T`, `u_U`, `omega_social`), each with
  a documented construct mapping and independence argument.
- **2 have no usable dataset on disk** (`eta_vol`, `gamma_sal`). External
  acquisition would require author contact or a source we have not verified.

Realistic ceiling for P6d: **10 synthetic priors → 2 synthetic priors**.
This is not "10 → 0". The 2 that remain require external data the project
does not have.

## 2. Definition of grounding

A prior is grounded if and only if:

1. An instrument measures the construct, or a documented proxy.
2. The instrument's source dataset is independent of every column currently
   in `work_g2/synthetic_32k.pkl`.
3. The construct-validity argument is written down, with a flag if the
   mapping is approximate.
4. Sample size N ≥ 100.

The failed P6d attempt at `1cc5e19` violated condition 2 for all 10 priors
by rank-transforming columns already in the copula. This scope imposes the
stricter rule to prevent recurrence.

## 3. Coverage matrix

| prior | active in engine? | on-disk candidate | source | N | type | verdict |
|---|---|:---|:---|:---:|:---:|:---|
| `m_t` | yes | PANAS pre-intervention | v5nk7 Study2B | 329 | **direct** | **GROUNDED-DIRECT** |
| `theta_T` | yes | HCP EMOTION FACE_ACC | 3T_tfMRI | 482 | proxy | **GROUNDABLE-PROXY** |
| `u_U` | yes | HCP GAMBLING PUNISH_PROP_LARGER | 3T_tfMRI | 491 | proxy | **GROUNDABLE-PROXY** |
| `omega_social` | yes | trust-game investment | XAI_Social_Games | ~200 | proxy | **GROUNDABLE-PROXY** (same-source caveat) |
| `eta_vol` | yes | none verified | — | — | — | **EXTERNAL-REQUIRED** |
| `gamma_sal` | yes | none useful on disk | — | — | — | **EXTERNAL-REQUIRED** |
| `lambda_norm` | **no** | MFQ in v5nk7 | — | — | — | **INERT-IN-ENGINE** |
| `s_spite` | **no** | none | — | — | — | **INERT-IN-ENGINE** |
| `tau_sens` | **no** | none | — | — | — | **INERT-IN-ENGINE** |
| `kappa_ctx` | **no** | none | — | — | — | **INERT-IN-ENGINE** |

## 4. The inert set — a finding that changes P6d

`work_env/p6c_v5.py` reads, from the parameter dict, exactly these
[SYNTHETIC-PRIOR] columns:

    theta_T     (3 occurrences)
    u_U         (3 occurrences)
    eta_vol     (3 occurrences)
    omega_social (1 occurrence)
    gamma_sal   (1 occurrence)
    m_t         (1 occurrence)

It does **not** reference `lambda_norm`, `s_spite`, `tau_sens`, or `kappa_ctx`
(zero occurrences, verified by grep at `p6d_archives_verify.log`).

Consequence: those four carry a `[SYNTHETIC-PRIOR]` flag through every
artifact but have no effect on any decision, any Q-value, any reward, or
any benchmark. Grounding them from MFQ-2 or any other source is
scientifically empty. They should be either:

- **retired from the copula** in the next rebuild (the 30-column matrix
  becomes 26 columns), or
- **explicitly flagged `[INERT-IN-CALIBRATED-ENGINE]`** in the G9 matrix
  and the XAI trace flags, so downstream consumers know the flag does
  not imply the parameter is load-bearing.

This scope recommends retirement. The current copula propagates a false
impression that 10 priors are structurally relevant when only 6 are.

## 5. Direct grounding — m_t from PANAS

**Source:** `v5nk7-osfstorage-archive/Study2B/Study2B_Data.csv`,
columns `panas_pos_pre` and `panas_neg_pre`.

**Instrument:** Positive and Negative Affect Schedule (Watson, Clark &
Tellegen, 1988). 20 items, 10 positive (PA) and 10 negative (NA), rated
on a 1–5 scale. Pre-intervention administration.

**Sample:** 401 rows total → 331 after `exclude == 0` → 329 after
`bot.detect == 0`. All 329 have finite PANAS pre scores.

**Construct:** mood valence at study entry = PA − NA. Directly matches the
engine's `m_t`, which is a per-agent constant influencing reward scaling
via `mult = omega * (1 + sal + 0.05 * mood)`.

**Distribution:**

    n = 329
    mean = 0.0000
    sd   = 1.0000
    min  = -2.5874
    max  = +2.1527
    q01..q99: [-2.31, -1.51, -1.25, -0.71, +0.01, +0.63, +1.28, +1.76, +2.15]

**Comparison with current copula:**

    q     PANAS     current    delta
    01   -2.3083   -2.2908   -0.0175
    05   -1.5141   -1.6457   +0.1315
    10   -1.2458   -1.2695   +0.0236
    25   -0.7092   -0.6697   -0.0395
    50   +0.0063   -0.0233   +0.0295
    75   +0.6323   +0.6567   -0.0244
    90   +1.2762   +1.2538   +0.0224
    95   +1.7592   +1.6495   +0.1097
    99   +2.1527   +2.3257   -0.1730

Largest deviation ±0.13. Within tolerance for the calibrated engine's
linear mood term (coefficient 0.05).

**Tail difference:** current copula has 0.25% of agents with |m_t| > 3.
PANAS-derived values clip at ±2.59 because N=329 does not contain extreme
individuals. Acceptable: the 0.05·mood contribution to the reward multiplier
is a 0.15 swing at the extremes, well within reward noise.

**Flags:**
- `[GROUNDED-DIRECT: PANAS-PRE]`
- `[STATE-TO-TRAIT-MAPPING]` — the engine treats `m_t` as a constant
  per agent; PANAS is a momentary state measure mapped to that constant.

**Extraction:** `Data_Xai32kProje/P6d_External/m_t_panas_pre.npy`,
n=329, mean 0, sd 1, clipped to [−3, 3].

## 6. Proxy grounding — theta_T, u_U, omega_social

### 6.1 theta_T ← HCP EMOTION FACE_ACC

- **Source:** `3T_TFMRI_inscanner_performance_data/*_tfMRI_EMOTION_*_Stats.csv`.
- **Instrument:** HCP Emotion task, condition FACE, measure ACC and MEDIAN_RT.
- **N:** 482 subjects.
- **Construct argument:** face-matching accuracy on fear/anger stimuli indexes
  threat-detection efficiency. The engine's `theta_T` weights the avoid/engage
  arbitration (`avd = theta_T + u_U`). Higher accuracy should map to lower
  threat sensitivity if the parameter is interpreted as a threshold, or to
  higher if it is interpreted as a signal. **Direction of the mapping must be
  fixed before use.** Recommendation: interpret `theta_T` as a threshold, so
  higher FACE_ACC → lower `theta_T`. Flag `[DIRECTION-ASSUMED]`.
- **Independence:** HCP 3T_tfMRI is not used by the current copula.

### 6.2 u_U ← HCP GAMBLING PUNISH_PROP_LARGER

- **Source:** `3T_TFMRI_inscanner_performance_data/*_tfMRI_GAMBLING_*_Stats.csv`.
- **Instrument:** HCP Gambling task, condition PUNISH, measure PROP_LARGER.
- **N:** 491.
- **Construct argument:** under punishment feedback, the proportion of
  larger-option choices is a risk-aversion index. `u_U` in the engine adds to
  threat in the avoid term. Loose mapping. Flag `[LOOSE-CONSTRUCT-MAPPING]`.
- **Independence:** as above.

### 6.3 omega_social ← XAI_Social_Games trust task

- **Source:** `XAI_Social_Games_Data/*_task-trust_*`.
- **Instrument:** trust-game investment ratio.
- **N:** ~200 across two file variants.
- **Construct argument:** investment in the trust game indexes valuation of
  social exchange. `omega_social` in the engine scales every reward.
- **Independence caveat:** `lambda_ineq` already draws from
  `XAI_Social_Games_Data` (the `ugr` task). Same subjects, different task.
  Flag `[SAME-SOURCE-AS-lambda_ineq]`.

## 7. External-required — eta_vol, gamma_sal

### 7.1 eta_vol

Requires a volatile bandit paradigm. On-disk candidates: none. OSF search
returned mostly stationary MAB experiments. The Gorilla "Learning Experiment"
(replication of Behrens 2007) shares only the task definition, not participant
data. **Requires external acquisition:** CORTEX database (unverified —
OSF API returned 403 to our tools), or author contact (Behrens, Niv,
Lawson, or any group running a volatile-bandit study).

### 7.2 gamma_sal

Requires an oddball/salience-capture paradigm with per-subject measures.
On-disk candidates: ERP CORE (OSF `thsqg`) contains raw EEG only.
Behavioural summaries, if they exist, are not accessible without downloading
multi-GB raw data. The archives we downloaded (`z3hfp`, `hckq7`) contain
filter scripts and ICA weights, not subject data.
**Requires external acquisition:** a study with salience behavioural summaries.

## 8. Path forward

**Immediate (Path 1 — P6d-Lite):**

Ground `m_t` from PANAS. Document `theta_T`, `u_U`, `omega_social` as
proxies. Retire the 4 inert columns from the copula. Produce
`synthetic_32k_p6d.pkl`. Verify `p6c_verify2.py` returns 5/5 on the new
copula. Effort: 1–2 sessions.

**Deferred (Path 2 — external):**

`eta_vol` and `gamma_sal` require datasets the project does not have.
Record them as `[EXTERNAL-REQUIRED]` in the G9 matrix. Do not attempt
further OSF searches without a specific, named dataset.

**Not a path:** rank-transforms of existing columns. The `[P6D-RANK-TRANSFORM-REJECTED]`
flag stays.

## 9. Decision required

Proceed with P6d-Lite?

- **Yes:** next block builds `synthetic_32k_p6d.pkl` with `m_t` direct,
  `theta_T`/`u_U`/`omega_social` as proxies, and the 4 inert columns removed.
  Then re-verify P6c.
- **Partial:** ground only `m_t`, leave the other 5 active priors as synthetic.
  Smallest change, largest pass probability, leaves 5 synthetics.
- **No:** document the analysis, freeze the current copula, publish with
  the limitation.
