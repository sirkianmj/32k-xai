# Delay discounting extraction — notes

**Source:** `Data_Xai32kProje/XAI_Delay_Discounting_Data/Delay Discounting Task_March 21, 2018.csv`
**Platform:** Qualtrics survey export, 476 non-preview respondents.
**Instrument:** Adaptive 5-trial titration task (McClure-style). Not a fixed
MCQ and not independent trials.

## Design (reconstructed from raw columns)

The survey presents 31 "Which would you rather have?" slots, but they are
**nodes in an adaptive binary decision tree**, not 31 independent trials.
Every respondent walks exactly 5 levels; the branch taken at each level
determines which node they see next. Evidence:

- Per-subject `n_answered` = 5 for every one of 476 subjects.
- Per-node `answered` counts chain through branches
  (e.g. `trial1 answered = 476`; `trial2 answered = SS(trial1) = 43`;
  `trial3 answered = LL(trial1) = 433`).

Root choice split: 43 SS / 433 LL.

## What the data gives us

- **`k value`** — per-subject discount rate, pre-computed by whoever ran the
  survey. Range 1.1e−4 to 24; median 6.7e−3; log-normal distribution.
- **`ED50`** — 1-to-1 with `k value` (same fit, different reporting).
- **Choice path** — recoverable from the raw columns as SS vs LL at each node.
- **`duration_s`** — total survey duration per subject.

## What the data does NOT give us

- **Actual dollar amounts** for SS and LL options per node. The Qualtrics
  template engine failed to substitute `Amount`/`Currency`/`Commodity`
  fields, so every row shows the unrendered string
  `${e://Field/Currency}${e://Field/Amount} in 3 weeks`.
- **Delays** per node. Visible strings include "in 3 weeks" and "in 2 years"
  but not every node; the full delay grid is not reconstructible.
- **Any shared subject ID** with HCP, IPIP, DASS, or SD3. Standalone sample.

Consequence: we **cannot refit `k` from this export**. We can only trust the
pre-computed column, which we validated against behavior (below).

## Empirical validation of the pre-computed `k`

| Test | Result |
| :--- | :--- |
| Median k, root-SS (n=43) | 0.189 |
| Median k, root-LL (n=433) | 0.0047 |
| Mean log k, root-SS | −1.51 |
| Mean log k, root-LL | −5.59 |
| Mann-Whitney U, SS k > LL k | U = 18619, p = 8.98e−28 |
| 75th pct of LL vs 25th pct of SS | 0.013 vs 0.082 (no overlap) |

**Monotone trend across n_SS levels** (n_SS = number of SS choices in 5-trial
path):

| n_SS | n subjects | mean log k |
| :--- | :--- | :--- |
| 0 | 16 | −9.115 |
| 1 | 72 | −6.485 |
| 2 | 162 | −5.750 |
| 3 | 203 | −4.368 |
| 4 | 22 | −2.598 |
| 5 | 1 | +3.178 |

Strictly monotonic for n_SS ∈ {0,1,2,3,4}, covering 475 of 476 subjects. This
is the textbook prediction of the hyperbolic discounting model and confirms
that the `k value` column is coherent with observed choice behavior.

## Quality filters applied

1. Drop `Distribution Channel == "preview"` (2 rows): 476 of 478 retained.
2. Require `k > 0` and finite: 476 retained (0 dropped).

## Outliers

One subject has k = 24 (log k = 3.18), 200× the 95th percentile (0.117). This
single subject is the only member of the n_SS = 5 group. It is a legitimate
extreme response, not a data error, but its leverage on distributional fits
should be noted. The `dd_k_subjects.csv` retains it; downstream users can
truncate at their discretion.

## Parameter dictionary status

- **k_DD**: extracted (476 subjects). Range > 0, log-normal.
  Provenance: pre-computed by survey author, validated by us against behavior.
  Flags: `[ADAPTIVE-DESIGN]` (5-trial tree, not 60-trial fixed), `[NO-REFIT]`
  (raw amounts unavailable).

## Sample characteristics

Median k = 0.0067, which corresponds to a mild-to-moderate discounting
phenotype. This is consistent with a self-selected adult online sample.

## Methodological honesty note

Two issues were discovered and corrected during extraction:

1. An initial script iteration used name-based indexing on columns that share
   the identical header string `"Which would you rather have?"`. Pandas
   returned a Series for each row rather than a scalar, producing silent NaN.
   Fixed with positional indexing (`df.iloc[:, i]`).
2. The expected sign of the `corr(k, n_SS)` diagnostic was initially stated
   as negative; the correct expectation is positive (higher k implies more
   SS choices). Observed r = +0.172, in the correct direction but weak in
   absolute terms — the stronger validation is the root-level Mann-Whitney
   test above.

## Reproducibility

Script: `score_dd.py`. Log: `score_dd.log`. Runtime: 0.1 s. Run with:
cd work_dd && python score_dd.py
