# SD3 extraction — notes

**Source:** `Data_Xai32kProje/SD3/data.csv`
**Platform:** Open Psychometrics. No participant ID; standalone sample.
**Instrument:** Short Dark Triad (Paulhus & Jones, 2011). 27 items, 9 per
subscale, rated on a 5-point Likert scale (1 = disagree, 5 = agree).
**Collection year (file metadata):** 2014.

## Item-level data preserved

The G3 bifactor CFA requires item-level responses, not just subscale sums.
Both are produced:

- `sd3_items_raw.csv` — 27 raw items, no reverse-keying (for CFA with method
  factor, if desired)
- `sd3_items_revkeyed.csv` — 27 items after reverse-keying {N2, N6, N8, P2, P7}
- `sd3_subscales.csv` — Mach / Narc / Psyc sums (range 9–45) plus normalized
  columns `*_n` in [0, 1]

The reverse-keyed convention follows Paulhus & Jones (2011). Narcissism items
N2, N6, N8 and psychopathy items P2, P7 are the standard reverse-keyed items.

## Quality filters applied

1. NaN filter (none triggered — no missing items).
2. Range filter: all 27 items must be integers in {1, 2, 3, 4, 5}.
   Dropped 452 rows (2.5%); 17,740 of 18,192 rows retained.

## Reliability (Cronbach's alpha, reverse-keyed)

| Subscale | alpha |
| :--- | :--- |
| Machiavellianism | 0.8606 |
| Narcissism | 0.8050 |
| Psychopathy | 0.7978 |

All three subscales exceed the 0.75 threshold commonly cited as acceptable
for research use on the SD3. The relative ranking (Mach > Narc > Psyc) matches
the published pattern in Paulhus & Jones (2011).

## Subscale correlations (raw)

|  | Mach | Narc | Psyc |
| :--- | :--- | :--- | :--- |
| Mach | 1.000 | 0.514 | 0.673 |
| Narc | 0.514 | 1.000 | 0.567 |
| Psyc | 0.673 | 0.567 | 1.000 |

All three subscales correlate positively and substantially (r = 0.51–0.67).
This pattern is the direct empirical motivation for a **bifactor specification**:
one general Dark Core (D) plus three specific factors (S_M, S_P, S_N). If the
three traits were truly orthogonal, these correlations would be near zero; if
they were fully explained by one factor, the CFA would show no distinct
specific loadings.

The G3 CFA will directly compare:
- (a) three correlated factors
- (b) one general factor only (unidimensional)
- (c) bifactor: general Dark Core + three specifics

Bifactor is expected to fit best on the basis of the Knitter et al. (2025)
meta-analytic SEM result on the SD3. That is a *hypothesis to be tested on this
sample*, not an assumption.

## Sample characteristics (caution)

Machiavellianism has a notably high mean (0.678 normalized), while psychopathy
has the lowest (0.454). This pattern is consistent with self-report SD3
literature: M items are the most socially acceptable to endorse, P items the
least. For the Vine Copula, the SD3 marginals from this sample capture the
*shape* of the distribution; the *location* is sample-specific and should be
adjusted if a normative anchor is required.

## References

- Paulhus, D. L., & Jones, D. N. (2011). Introducing a short measure of the
  Dark Triad. Poster, Society for Personality and Social Psychology, San Antonio.
- Knitter, L. A., Hoffmann, J., Eid, M., & Koch, T. (2025). Measuring the dark
  triad: A meta-analytical SEM study of two prominent short scales.
  Frontiers in Psychology, 15, 1469970.

## Reproducibility

Script: `score_sd3.py`. Log: `score_sd3.log`. Runtime: 0.2 s. Run with:
cd work_sd3 && python score_sd3.py

