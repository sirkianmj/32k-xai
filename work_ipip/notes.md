# IPIP-FFM OCEAN extraction — notes

**Source:** `Data_Xai32kProje/IPIP-FFM-data-8Nov2018/data-final.csv`
**Platform:** Open Psychometrics (public release, 8 Nov 2018).
**Instrument:** IPIP-50 Big Five Factor Markers (Goldberg).
**Collection:** 2016-2018, self-selected online sample.

## Quality filters

1. NaN filter: drop rows with any missing item (1,783 of 1,015,341 removed).
2. Range filter: all 50 items must be integers in {1, 2, 3, 4, 5}. 139,124 rows
   had at least one out-of-range response and were removed.
3. `IPC == 1` filter (cleaner subset): keep only rows from an IP address
   with a single submission. 603,322 of 874,434 rows retained.

## Parameters extracted

- **O** (Openness) — mean 0.735, SD 0.155
- **C** (Conscientiousness) — mean 0.586, SD 0.185
- **E** (Extraversion) — mean 0.479, SD 0.228
- **A** (Agreeableness) — mean 0.690, SD 0.184
- **N** (Neuroticism) — mean 0.520, SD 0.215

Range: each normalized to [0, 1] via `(sum − 10) / 40`.

## Correlation matrix

| | O | C | E | A | N |
| :--- | ---: | ---: | ---: | ---: | ---: |
| O | 1.000 | 0.055 | 0.175 | 0.102 | −0.079 |
| C | 0.055 | 1.000 | 0.068 | 0.137 | −0.235 |
| E | 0.175 | 0.068 | 1.000 | 0.306 | −0.225 |
| A | 0.102 | 0.137 | 0.306 | 1.000 | −0.040 |
| N | −0.079 | −0.235 | −0.225 | −0.040 | 1.000 |

All correlations are in the expected directions for the IPIP-50.

## Anti-double-counting

Neuroticism (N) is used as the OCEAN dimension. The DASS general distress
factor is a *distinct* construct; DASS subscales are secondary modulators.
No negative affectivity parameter is separately included in the dictionary.

## Flags

- `[OPEN-PSYCHOMETRICS]`
- `[SELF-SELECTED]` — mean-level O and A are higher than normative samples.

## Reproducibility

Script: `score_ipip.py`. Log: `score_ipip.log`. Runtime: ~8 s. Run with:
cd work_ipip && python score_ipip.py

