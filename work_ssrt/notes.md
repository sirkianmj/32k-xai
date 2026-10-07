# Stop-signal SSRT extraction — notes

**Source:** `Data_Xai32kProje/XAI_Risk_Impulsivity_Data/sub-*_task-stopsignal_events.tsv`
**Subjects:** 266 files processed, 0 skipped.
**Design:** adaptive 1-up/1-down tracking procedure, 50 ms step size
(inferred from data — see §Design reconstruction). Each subject: 96 GO
trials and 32 STOP trials.

## Method

SSRT estimated via the **integration method with replacement of go
omissions**, per Verbruggen et al. (2019, eLife, Recommendation 8).

    SSRT = nth_go_RT − mean_SSD
    nth = round((1 − p(respond|signal)) × N_go_including_replacements)

Where `p(respond|signal)` = UnsuccessfulStop / all STOP trials. Where GO
omissions are present, each is replaced by the maximum GO RT in the
subject's trial set (Rec. 8; Materials and Methods, Verbruggen et al. 2019).

The integration method is the recommended default for the adaptive
tracking procedure. The paper's simulations (Appendix 2) report that this
method is the least biased and most reliable of the non-parametric methods
(overall r = 0.57 vs. r = 0.53 for the mean method; r = 0.51 for the
integration method without replacement; r = 0.43 for integration with
adjusted p(respond|signal)).

## Design reconstruction

The adaptive nature of the task is visible in the raw data:

- 1-up/1-down: SSD increases after SuccessfulStop, decreases after
  UnsuccessfulStop. Step size in the data is a mix of ~50 ms and
  variable values consistent with a staircase that has not yet
  converged in the first few trials (SSD values observed as low as 0).
- SSD = 0 trials (up to ~6 per subject) are generated naturally by the
  staircase. They are NOT diagnostic trigger-failure trials of the kind
  described in Box 2 of Verbruggen et al. (2019). The paper instructs
  the exclusion of deliberately inserted diagnostic SSD = 0 trials only;
  it is silent on staircase-generated SSD = 0 trials.

## Quality filters applied

1. GO filter: successful GO only, RT in [100 ms, 1500 ms]. GO choice
   errors were detected (`UnuccessfulGo`) but are rare in this dataset
   (0 per subject on average); they are retained in the responded-GO RT
   pool per Rec. 8.
2. GO omissions (RT < 100 ms or missing) were replaced with the subject's
   maximum valid GO RT, per Rec. 8.
3. Minimum trial counts: ≥ 30 responded GO trials and ≥ 15 STOP trials.
4. Endpoint clipping: p(respond|signal) values of exactly 0 or 1 are
   excluded from SSRT estimation (the nth-percentile formula is undefined
   at the endpoints).
5. Race-model check (Rec. 7): subjects with mean RT(UnsuccessfulStop) >
   mean RT(responded GO) are flagged `RACE_VIOLATION`. Per the paper this
   check is diagnostic, not necessarily exclusionary; flagged subjects
   are retained in the CSV.

## Results

| Metric | Canonical `ssrt` (with replacement) |
| :--- | :--- |
| Subjects processed | 266 |
| SSRT computable | 264 |
| Mean | 0.2338 s |
| **Median** | **0.2174 s** |
| IQR | 0.1851 – 0.2644 s |
| SD | 0.0782 s |
| Min / Max | 0.1113 / 0.9437 s |
| SSRT ≤ 0 | 0 |

Sensitivity variants (all retained in the CSV, none used as canonical):

| Variant | Mean | Median | Max | SSRT ≤ 0 |
| :--- | :--- | :--- | :--- | :--- |
| `ssrt` (with replacement) | 0.2338 | 0.2174 | 0.9437 | 0 |
| `ssrt_noreplace` | 0.2279 | 0.2171 | 0.4228 | 0 |
| `ssrt_pos` (SSD > 0 only) | 0.2076 | 0.1954 | 0.8795 | 1 |

Pairwise correlations:

- `corr(ssrt, ssrt_noreplace) = 0.8965`
- `corr(ssrt, ssrt_pos)       = 0.9168`

## Outlier in the canonical estimator

**One subject has `n_go_omissions = 46` out of ~96 GO trials** — nearly
half of all GO trials missed. Because the with-replacement rule assigns
this subject's maximum GO RT to each of 46 omitted trials, the nth-RT
is pushed upward and the resulting SSRT is 0.9437 s (max in sample).
Without replacement the same subject's SSRT is 0.4228 s.

This is the correct behavior of the recommended estimator, not a bug.
But the outlier inflates the canonical SD and tail. Two remedies are
documented for downstream:

- Exclude subjects with `n_go_omissions / n_go_total > 0.20` (a
  conservative omission threshold); this is the approach Verbruggen et
  al. discuss in Box 2 for high-omission subjects.
- Or use the median SSRT (0.2174 s) as the population anchor rather than
  the mean (0.2338 s).

The CSV retains all 266 subjects so the exclusion decision can be made
explicitly at the CFA / copula stage.

## Choice of `ssrt` (with replacement) as canonical over `ssrt_pos`

`ssrt_pos` post-hoc excludes SSD = 0 ms trials from the mean-SSD term.
Three reasons make this inappropriate for this dataset:

1. **The SSD = 0 trials are valid tracking trials.** They were generated
   by the adaptive staircase, not deliberately inserted as diagnostic
   probes. Verbruggen et al. (2019, Box 2) instructs the exclusion of
   only the latter. The paper is silent on staircase-generated SSD = 0
   trials; the default is therefore to include them.

2. **The exclusion changes the estimator in a subject-specific way.**
   Mathematically, mean_ssd_pos ≥ mean_ssd_all, so `ssrt_pos ≤ ssrt`
   whenever the nth-RT is unchanged. The offset per subject depends on
   how many SSD = 0 trials that subject happened to receive during
   staircase excursions. This is a **post-hoc truncation of the SSD
   distribution**, not a systematic bias correction.

3. **`ssrt_pos` produces an implausible negative value (−12 ms) in one
   subject.** A negative SSRT is not physically interpretable under the
   race model. This is a confirmatory signal, not the primary
   justification — the primary justification is (1) and (2) above.

`ssrt_pos` is retained in the CSV as a sensitivity variant only.

## Deviations from the recommended method

Two earlier deviations from Verbruggen et al. (2019) were corrected in
this revision of `score_ssrt.py`:

1. **GO omissions are now replaced** with the subject's max GO RT
   (previously dropped). Per Rec. 8.
2. **Race-model check added** (Rec. 7): subjects with mean RT on
   unsuccessful STOP trials > mean RT on responded GO trials are flagged
   `RACE_VIOLATION`. There are 14 such subjects (5.3% of the sample).
   Their SSRTs are retained in the CSV; downstream analyses may exclude
   them.

## Race-model check results

14 subjects (5.3%) triggered `RACE_VIOLATION` under the Rec. 7
comparison. Verbruggen et al. (2019, Appendix 2) report that apparent
violations are common even when the underlying race model holds — the
comparison is a diagnostic, not a definitive exclusion criterion. The
flagged subjects are retained in the CSV.

## Flag breakdown (all subjects, any flag)

| Flag | Count |
| :--- | :--- |
| RACE_VIOLATION | 14 |
| P_HIGH (p_respond > 0.80) | 6 |
| FEW_GO (< 30 responded GO) | 1 |
| SSRT = NaN | 2 |

## Parameter dictionary status

- **a_inhib** (`SSRT index`): extracted (264 valid of 266 subjects).
  Units: seconds. Interpretation: lower SSRT = better inhibitory control.
  Median = 0.2174 s; mean = 0.2338 s (inflated by one high-omission
  subject; see Outlier).
  Provenance: integration-method computation from trial-level RT data
  (G7 confirmed RTs are present).
  Flags: `[HCP-PROCEDURE]` (staircase, 96 GO / 32 STOP),
  `[OMISSION-REPLACED]` (Rec. 8 implemented),
  `[RACE-CHECKED]` (14 subjects flagged, retained).

## Reproducibility

Script: `score_ssrt.py`. Log: `score_ssrt.log`. Runtime: 1.7 s.
cd work_ssrt && python score_ssrt.py

