# Stop-signal SSRT extraction — notes

**Source:** `Data_Xai32kProje/XAI_Risk_Impulsivity_Data/sub-*_task-stopsignal_events.tsv`
**Subjects:** 266 files processed, 0 skipped.
**Instrument:** HCP-style stop-signal task with adaptive staircase
(`LadderNumber`, `LadderMovement` columns). Each subject: 96 GO trials +
32 STOP trials.

## Method

SSRT computed via the **integration method** (Verbruggen et al., 2019, eLife):

    SSRT = nth_GO_RT − mean_SSD
    where n = round((1 − P_respond|signal) × N_go_correct)

`nth_GO_RT` is the nth percentile of the RT distribution of correct GO
trials, and `P_respond|signal` is the fraction of STOP trials on which the
subject responded (i.e., UnsuccessfulStop / all STOP trials).

## Quality filters applied

1. **GO RT filter:** correct GO trials only (`TrialOutcome == "SuccessfulGo"`),
   with RT in [100 ms, 1500 ms]. Excludes anticipatory and inattentive
   responses.
2. **Minimum trial counts:** ≥ 30 valid GO RTs and ≥ 15 STOP trials for
   SSRT computability.
3. **Endpoint clipping:** `P_respond` values of exactly 0 or 1 are excluded
   from SSRT computation, as they make the nth-percentile formula undefined.

Flagging thresholds follow Verbruggen et al. (2019):
- `P_LOW`: P_respond < 0.20 (staircase too easy — subject rarely fails stop)
- `P_HIGH`: P_respond > 0.80 (staircase too hard — subject almost always fails stop)
- `FEW_GO`: fewer than 30 valid GO RTs
- `FEW_STOP`: fewer than 15 STOP trials

## Results

| Metric | Value |
| :--- | :--- |
| Subjects processed | 266 |
| SSRT computable | 264 |
| Flagged (any reason) | 7 |
| Median SSRT | 0.2171 s |
| Mean SSRT | 0.2279 s |
| IQR SSRT | 0.1849 – 0.2605 s |
| Range | 0.1113 – 0.4228 s |
| Mean P_respond | 0.5191 |
| Mean GO RT | 0.5178 s |
| Mean SSD | 0.2711 s |

**Distributional check:** median SSRT = 217 ms sits squarely within the
200–300 ms range reported for healthy young adults in the stop-signal
literature. Zero subjects have SSRT ≤ 0; zero have SSRT > 1000 ms. Both
are expected patterns for a well-behaved staircase.

## Excluded / flagged subjects

- **2 subjects** have SSRT = NaN:
  - 1 flagged `FEW_GO` (subject with zero valid GO RTs)
  - 1 with P_respond = 1.0 (failed every stop trial — endpoint exclusion)
- **6 subjects** flagged `P_HIGH` (P_respond > 0.80): they responded on more
  than 80% of stop trials, indicating the staircase did not slow them enough.
  Their SSRT values are retained in the CSV but the flag is documented. The
  Vine Copula stage may choose to exclude these if they distort tail behavior.

## Methodological divergence (documented)

Two SSRT variants were computed:

- **`ssrt`** — mean SSD across all STOP trials (including SSD = 0)
- **`ssrt_pos`** — mean SSD across STOP trials with SSD > 0 only

**Correlation between the two variants: r = 0.8778.** This is below the 0.90
threshold declared before observing the data. The difference is not noise.

| Variant | Mean | Min | Max | Implausible values |
| :--- | :--- | :--- | :--- | :--- |
| ssrt (all SSD) | 0.2279 | 0.1113 | 0.4228 | 0 |
| ssrt_pos (SSD>0) | 0.2017 | −0.0122 | 0.4037 | 1 negative |

**Canonical choice: `ssrt` (all SSD).** Justification:

1. Verbruggen et al. (2019) explicitly recommend including stop trials
   with SSD at the minimum (here 0), as they contribute valid information
   about the race between go and stop processes.
2. The all-SSD variant yields zero implausible values; the positive-SSD
   variant yields one negative SSRT.
3. Systematic offset of ~26 ms between variants is attributable to
   subject-level variance in the number of SSD=0 staircase warm-up trials.

`ssrt_pos` is retained in the output CSV as a sensitivity check but is **not**
the canonical parameter.

## Parameter dictionary status

- **a_inhib** (`SSRT index`): extracted (264 valid of 266 subjects).
  Units: seconds. Interpretation: lower SSRT = better inhibitory control.
  Provenance: our integration-method computation from trial-level RT data
  (G7 confirmed RTs are present).
  Flags: `[HCP-PROCEDURE]` (staircase task, not fixed-SSD),
  `[NEG-CONTROL-CHECKED]` (all SSRT > 0 in canonical variant).

## Sample characteristics

Mean GO RT = 518 ms, mean SSD = 271 ms, mean P_respond = 0.519. All three
are consistent with a well-calibrated adaptive staircase at the standard
50% inhibition target. This is a high-quality dataset for SSRT.

## Reproducibility

Script: `score_ssrt.py`. Log: `score_ssrt.log`. Runtime: 1.1 s. Run with:
cd work_ssrt && python score_ssrt.py
