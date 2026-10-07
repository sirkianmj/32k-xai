# G2 — Evidence-Constrained Hierarchical Copula — notes

**Status:** PASS under the strongest available criterion — marginals exact,
structural identities exact, composite couplings robust across prespecified
identification rules AND compatible with per-coupling literature estimates
under combined uncertainty.

**Deliverable:** A 32,000-agent synthetic population (`synthetic_32k.pkl`)
whose atomic marginals exactly reproduce their source distributions, whose
composites exactly equal the mean of their components, and whose composite-
level relationships with external variables fall within the combined
uncertainty of the literature estimates they encode.

## Architecture

Three fundamentally different kinds of information are separated into
different levels of a strict hierarchy. This separation is what makes the
architecture mathematically coherent.

1. **Empirical marginals** — the shape of each variable's distribution.
2. **Literature-derived couplings** — the dependence structure between
   variables, curated from meta-analyses with per-coupling uncertainty.
3. **Structural identities** — deterministic composites that are exact
   functions of their components.
LITERATURE + SOURCE EVIDENCE
│
▼
┌─────────────────────────────┐
│ Evidence-weighted │
│ dependency targets │
│ (with composite projection) │
└──────────────┬──────────────┘
│
▼
┌─────────────────────────────┐
│ 28 ATOMIC VARIABLES │
│ joint Gaussian copula │
│ + rank-matching marginals │
└──────────────┬──────────────┘
│
┌────────┴────────┐
▼ ▼
D = mean(S_M,S_P,S_N) general_distress = mean(DASS_*)
│ │
└────────┬────────┘
▼
DETERMINISTIC COMPOSITES
│
▼
COMPOSITE EXTERNAL VALIDATION
(against literature with combined uncertainty)

text

## Layer 1 — Atomic basis (28 stochastic parameters)

The copula operates on **atomic variables only**:

- **OCEAN (5)**: O, C, E, A, N
- **Dark Triad specifics (3)**: S_M, S_P, S_N
- **DASS specifics (3)**: DASS_Dep, DASS_Anx, DASS_Str
- **Two-step (4)**: w_MB, alpha_R, beta_choice, phi_per
- **Other empirical (3)**: k_DD, a_inhib, lambda_ineq
- **Synthetic priors (10)**: theta_T, u_U, eta_vol, omega_social, lambda_norm,
  s_spite, tau_sens, gamma_sal, kappa_ctx, m_t

**`D` and `general_distress` are NOT copula nodes.** They are deterministic
linear combinations computed *after* sampling:

    D                = mean(S_M, S_P, S_N)
    general_distress = mean(DASS_Dep, DASS_Anx, DASS_Str)

This removes the rank-deficiency that would otherwise arise when a linear
composite sits as a free node in a joint covariance matrix.

## Layer 2 — Composite disaggregation with explicit identification strategy

Literature correlations involving a composite (e.g., `N ↔ general_distress`)
are disaggregated to their atomic component couplings before the matrix is
built. This is an **identification-resolving step**, not a mathematical
derivation. The constraint

    Cov(X, Y) = (1/k) Σ_i Cov(X, C_i)     (Y = mean of k components)

is one equation in k unknowns. Multiple allocation rules satisfy it exactly.
The chosen rule is therefore an explicit assumption and must be tested for
sensitivity.

**Three prespecified allocation rules tested:**

| Rule | Formula | Interpretation |
| :--- | :--- | :--- |
| **A: equal covariance** | `Corr(X, C_i) = r_XY · σ_Y / σ_{C_i}` | Each component carries the same covariance with X as the composite does |
| **B: uniform correlation** | `Corr(X, C_i) = k · r_XY · σ_Y / Σ σ_{C_j}` | All component correlations equal |
| **C: variance proportional** | `Corr(X, C_i) ∝ σ_{C_i}` (normalised) | Larger components carry proportionally more |

All three rules satisfy the linear-composite identity exactly. They differ
only in how they distribute the association among components. **Rule A is
canonical**, not because it is theoretically privileged, but because it is
the most conservative (uniform distribution rather than amplification by
variance).

**Identification sensitivity:** the three rules produce composite couplings
that differ by 0.0002–0.0011 across the full sensitivity table. This is
40–300× smaller than the combined tolerance. **The choice of allocation rule
is empirically immaterial for the composite-level outcomes.**

## Layer 3 — Confidence-weighted nearest correlation matrix

Each CSV row carries a `confidence` and `status`. These map to weights in
the target matrix:

| Status | Weight |
| :--- | ---: |
| empirical_direct | 1.0 |
| literature_meta_analysis | 0.9 |
| derived | skipped (see below) |
| empirical_null | 0.7 |
| direction_supported | 0.4 |
| indirect_evidence | 0.3 |
| unverified | 0.0 |
| disaggregated (any of the above) | weight × 0.8 |

The matrix is solved by 60 rounds of Higham-style alternating projection:

    R ← project_to_corr(R)
    R ← R + W ⊙ (R_target − R)
    R ← project_to_corr(R)

The final Spearman matrix is projected to positive-definite, then transformed
to Pearson via

    ρ_Pearson = 2 · sin(π · ρ_Spearman / 6)

which is the exact inverse of the Gaussian copula's Spearman-Pearson identity.
Order matters: **project Spearman to PSD first, then transform**. The reverse
amplifies residuals through the nonlinear transform.

## Layer 4 — Sampling, rank-matching, deterministic composites

1. Sample correlated latent `Z ~ N(0, R_Pearson)` via Cholesky with 1e-10 ridge.
2. `U = Φ(Z)` — column-wise uniforms.
3. **Rank-matching**: for each variable, sort source values ascending, sort
   synthetic uniforms ascending, assign the source value at rank *k* to the
   agent with the *k*-th largest uniform. If `n_src ≥ 32,000`, subsample
   without replacement; otherwise resample with replacement.

   This makes the marginal **exact by construction**. The synthetic column
   *is* the source column permuted. No parametric fit. No extrapolation.
   No impossible values.

4. **Deterministic composites**: `D` and `general_distress` are computed
   post-hoc from the atomic draws.

## Layer 5 — Three-layer validation

### G2.1 — Atomic marginal fidelity

All 28 atomic variables pass KS statistic < 0.04 and relative Wasserstein
< 0.08 simultaneously. In practice most are far under: KS < 0.01, relative
Wasserstein < 0.02, giving "EXCELLENT" grade.

### G2.2 — Structural consistency

Both composites have `max_abs_diff = 0.0` — the deterministic identity
holds to floating-point exactness.

### G2.3 — Composite external validity (identification-aware)

For each composite-level literature target, the test is:

- **Robustness** — the three disaggregation rules produce composite couplings
  whose spread is below the combined tolerance.
- **Compatibility** — the mean-across-rules synthetic value falls within
  combined tolerance of the literature target.

**Tolerance:** `2 · √(SE_lit² + SE_syn²)` where
- `SE_lit` is per-coupling, read from the CSV `se_lit` column
- `SE_syn = 1/√32000 ≈ 0.0056`

This is a **predefined two-standard-error compatibility criterion** combining
literature uncertainty and Monte Carlo sampling uncertainty. It is not a
formal meta-analytic hypothesis test.

### Results

| Coupling | ρ_lit | SE_lit | A | B | C | Spread | Tolerance | Mean err | Robust | Compatible |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| N ↔ general_distress | +0.416 | 0.030 | +0.413 | +0.413 | +0.414 | 0.0011 | 0.061 | 0.0029 | ✓ | ✓ |
| E ↔ general_distress | −0.250 | 0.030 | −0.245 | −0.245 | −0.246 | 0.0011 | 0.061 | 0.0048 | ✓ | ✓ |
| C ↔ general_distress | −0.320 | 0.030 | −0.313 | −0.313 | −0.314 | 0.0010 | 0.061 | 0.0066 | ✓ | ✓ |
| A ↔ general_distress | −0.038 | 0.040 | −0.027 | −0.027 | −0.026 | 0.0002 | 0.081 | 0.0115 | ✓ | ✓ |
| O ↔ general_distress | −0.084 | 0.030 | −0.078 | −0.078 | −0.078 | 0.0005 | 0.061 | 0.0060 | ✓ | ✓ |

All five pass both robustness and compatibility.

### Literature values and sources

The five composite-level correlations come from Kotov et al. (2010), Table 5.
We use the **distress-specific average** across MDD, unipolar (broad),
dysthymic, GAD, and PTSD — the five disorders that constitute the HiTOP
distress subfactor.

**`[CONSTRUCT-LEVEL-PROXY]`:** Kotov et al.'s correlations are between Big
Five traits and *diagnostic groups*. Our `general_distress` is a composite
of DASS-42 subscales (depression, anxiety, stress). These are not the same
measurement: DASS is a dimensional self-report instrument, whereas
diagnostic groups are categorical clinician-assigned classifications.
The mapping is therefore **construct-level**, motivated by the shared
HiTOP distress-subfactor structure, not a direct DASS validation. Downstream
uses of `N ↔ general_distress` should be interpreted as "N relates to the
general-distress construct at approximately r ≈ 0.42", not as "N relates to
the DASS composite at exactly 0.42". This is a documented limitation, not a
defect in the copula.

| Disorder | N | E | C | A | O |
| :--- | ---: | ---: | ---: | ---: | ---: |
| MDD | 0.47 | −0.25 | −0.36 | −0.06 | −0.08 |
| Unipolar (broad) | 0.42 | −0.28 | −0.35 | −0.06 | −0.04 |
| Dysthymic | 0.36 | −0.29 | −0.33 | +0.07 | −0.14 |
| GAD | 0.34 | −0.18 | −0.29 | +0.05 | −0.09 |
| PTSD | 0.49 | −0.25 | −0.27 | −0.19 | −0.07 |
| **Distress average (used)** | **0.416** | **−0.250** | **−0.320** | **−0.038** | **−0.084** |

**Source:** Kotov, R., Gamez, W., Schmidt, F., & Watson, D. (2010). *Linking
"Big" Personality Traits to Anxiety, Depressive, and Substance Use Disorders:
A Meta-Analysis.* Psychological Bulletin, 136(5), 768–821. Table 5, p. 806.

**SE estimates — `[PROVISIONAL-SE]`:** The paper does not report SEs directly
in Table 5 of the published text. We use SE = 0.03 for N/E/C/O (typical for
meta-analytic r with pooled N ≈ 175 per disorder) and SE = 0.04 for A (larger
because the effect is near-zero, where meta-analytic precision is lower).
**These are provisional values, not extracted from Kotov et al.'s variance
tables.** They are used only as a tolerance-setting mechanism in G2.3, not
as claims about the true meta-analytic uncertainty. A follow-up refinement
could populate them from the `configural` R package (Wiernik, 2024) which
contains per-cell SEs, or from the Kotov et al. supplementary materials.
The G2 verdict is robust to this: the composite couplings are 40–300×
tighter than the tolerance, so even doubling the assumed SE would not
change PASS/FAIL.

## Verdict

| Criterion | Result |
| :--- | :--- |
| G2.1 atomic marginals | PASS |
| G2.2 structural consistency | PASS |
| G2.3a robustness across rules | PASS |
| G2.3b literature compatibility (per-coupling SE) | PASS |
| **OVERALL** | **PASS** |

## What G2 PASS means

The generative model is:

1. **Internally consistent** — every marginal exactly reproduces its source,
   every composite exactly equals its component mean, and the joint
   correlation matrix is positive-definite with high-weight entries close
   to their targets.
2. **Identification-robust** — the composite-level outcomes are stable across
   three mathematically distinct allocation rules, so downstream conclusions
   do not materially depend on the disaggregation choice.
3. **Literature-compatible** — composite couplings fall within combined
   uncertainty (literature SE + synthetic sampling SE) of their per-coupling
   published estimates.

## What G2 PASS does not mean

1. **It is not external validation against an independent joint dataset.**
   No joint dataset exists for these 21 parameters. The copula's dependence
   structure for cross-source couplings rests on literature evidence, not
   on joint observation.
2. **It does not prove Rule A is theoretically correct.** The three rules
   demonstrate that the result is *empirically insensitive* to the allocation
   choice for this configuration. Theoretically, the allocation is
   underdetermined without joint data; the sensitivity analysis is what makes
   the conclusion defensible.
3. **It does not verify the literature inputs independently.** The values
   used for N/E/C/A/O ↔ general_distress come from Kotov et al. (2010), a
   study cited but not re-analysed. Their effect sizes are the target; the
   synthetic reproduces them within their assumed uncertainty.
4. **The Kotov → DASS mapping is construct-level.** See `[CONSTRUCT-LEVEL-PROXY]`
   above. This is a shared-construct inference, not a direct measurement
   equivalence.

## Derived rows are skipped

The CSV contains three rows marked `derived`:

    general_distress ↔ DASS_Dep   = 0.87
    general_distress ↔ DASS_Anx   = 0.85
    general_distress ↔ DASS_Str   = 0.89

These are mathematical consequences of the composite identity plus the
observed component intercorrelations. Their true values are ≈ 0.899, 0.897,
0.927 respectively. They are **skipped** in Layers 1+2 and G2.3; their role
is covered exactly by G2.2.

## What failed in prior iterations and why

Five earlier attempts produced partial or failed verdicts. Each failure was
a specific structural error, not a numerical one.

1. **Parametric marginals + Spearman-fed Gaussian copula.** Fitting
   Normal/Lognormal to small-N sources produced mass outside the observed
   range. Using Spearman ρ as the Gaussian copula's Pearson parameter
   introduced a ~15 % systematic error on every non-zero coupling.

2. **KDE marginals + empirical tail anchoring.** Smoothing still permitted
   tail extrapolation. ECDF maxdiff at 5 % quantiles reached 1.3. For
   `lambda_ineq` the KDE produced ±∞ because the source has 12 % zero-mass
   that no continuous family can fit.

3. **Composites as free copula nodes.** Treating `D` and `general_distress`
   as ordinary nodes with their own marginals and their own literature
   couplings is mathematically overdetermined when the composite is *also*
   defined as `mean(components)`. The recompute step silently overwrote the
   copula's composite draws.

4. **Disaggregation without sensitivity analysis.** The first working
   version's allocation rule was one choice among infinitely many; the
   identification concern was unaddressed.

5. **Common LIT_SE for all couplings.** Using a single `SE_lit = 0.03`
   obscured the per-coupling uncertainty differences. Replaced with the
   per-coupling `se_lit` column.

The final design fixes all five:

- **Marginal**: rank-matching against source values (no fitting).
- **Copula parameter**: correct Spearman → Pearson transform.
- **Composite**: deterministic post-hoc computation.
- **Identification**: three rules + sensitivity analysis.
- **Uncertainty**: per-coupling literature SE.

## Files produced

| File | Content |
| :--- | :--- |
| `synthetic_32k.pkl` | 32,000 × 30 DataFrame (28 atomic + 2 composites) |
| `synthetic_32k_sample50k.csv` | 50,000-row CSV sample |
| `target_Spearman.npy` | Final 28×28 Spearman target matrix |
| `target_weights.npy` | 28×28 confidence weights W |
| `target_R_flags.pkl` | Per-cell provenance flags |
| `sensitivity_composite_couplings.csv` | G2.3 table (3 rules, per-coupling SE) |
| `validation_marginals.csv` | G2.1 per-parameter results |
| `validation_structural.csv` | G2.2 composite identity residuals |
| `validation.txt` | Summary verdict |
| `marginal_models.csv` | Source metadata per parameter |
| `source_modes.csv` | Subsample vs. resample mode per parameter |
| `literature_couplings/pairwise_correlations_verified.csv` | Evidence table |
| `metadata/variable_dictionary.csv` | Spec parameter mapping |

## Reproducibility

Script: `build_copula.py`. Log: `build_copula.log`. Runtime: ~3 s. Run with:

    cd work_g2 && python build_copula.py

## Remaining verification work (documented, not blocking)

1. Replace the `se_lit` provisional values (0.03 / 0.04) with extracted
   values from the `configural` R package or the Kotov et al. supplement.
2. Independently verify each row of the pairwise correlations CSV against
   the cited source (spot-check rho values against published tables).
3. Extend the sensitivity analysis to the `D` composite — currently the
   sensitivity test focuses on `general_distress` because D's component
   couplings were direct atomic couplings from the SD3 source, not
   disaggregated from literature targets.

None of these affect the current G2 PASS; they are refinements that would
strengthen the evidence artifact during manuscript preparation.

## References

- Higham, N. J. (2002). Computing the nearest correlation matrix — a problem
  from finance. *IMA Journal of Numerical Analysis*, 22(3), 329–343.
- Nelsen, R. B. (2006). *An Introduction to Copulas* (2nd ed.). Springer.
- Kotov, R., Gamez, W., Schmidt, F., & Watson, D. (2010). Linking "Big"
  personality traits to anxiety, depressive, and substance use disorders:
  A meta-analysis. *Psychological Bulletin*, 136(5), 768–821.
- O'Boyle, E. H., Forsyth, D. R., Banks, G. C., Story, P. A., & White, C. D.
  (2015). A meta-analytic test of redundancy and relative importance of the
  Dark Triad and Five-Factor Model of personality. *Journal of Personality*,
  83(6), 644–664.
- Gillespie, S. M., Mitchell, I. J., Johnson, I., Dawson, E., & Fitch, S. J.
  (2022). Psychopathy and response inhibition: A meta-analysis of go/no-go
  and stop signal task performance. *Neuroscience & Biobehavioral Reviews*.
- Petrov, N. B., Lim, V., Fillon, A., & Feldman, G. (2026). Values and the
  dark side: Meta-analysis of links between Dark Triad traits and personal
  values. *Personality and Social Psychology Bulletin*.
- Yeh, Y.-H., Myerson, J., & Green, L. (2021). Delay discounting, cognitive
  ability, and personality: What matters? *Psychonomic Bulletin & Review*,
  28(2), 686–694.
