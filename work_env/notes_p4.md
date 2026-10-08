# work_env — Phase 4 notes — provenance propagation and k_DD decode

**Commit:** `G-ENV-P4` — synthetic-prior flags propagated, k_DD decoded.
**Date:** 08 October 2026.
**Hardware:** GTX 1660 Ti (6 GB VRAM), WSL2 Ubuntu, JAX 0.11.2 CUDA 13.

## Deliverable

- `engine_v5.py` patched: `PARAM_PROVENANCE` dict + `SYNTHETIC_PRIOR_PARAMS`
  list, canonical k_DD gamma.
- `xai_trace.py` patched: canonical k_DD gamma, provenance block in return.
- `run_32k_v5.py` patched: provenance written to summary file.
- Backup of pre-fix versions: `engine_v5.py.pre_p4fix`, `xai_trace.py.pre_p4fix`,
  `run_32k_v5.py.pre_p4fix`.

## Fix 1 — k_DD decode (was [COPULA-DEFECT-k_DD])

**Root cause:** `work_g2/build_copula.py` line 72:
    S["k_DD"] = dd["log_k"].to_numpy()

The copula column is log(k), not k. The spec §3.5 requires k > 0
hyperbolic discount rate.

**Evidence chain:**
- raw delay-discounting k: n=478, min=0.000110, mean=0.102445, max=24.0
- log(k): n=478, min=-9.1150, mean=-5.2271, max=3.1781
- copula k_DD column: min=-9.1150, mean=-5.2073, max=3.1781

Exact match. The copula is log(k_DD).

**Fix:** exponentiate the copula column to recover the true positive k:

    k_DD_true = jnp.exp(jnp.clip(P['k_DD'], -10.0, 5.0))
    gamma     = 1.0 / (1.0 + k_DD_true)

This is the canonical hyperbolic-discount transform per spec §2.5.
The clip bounds [−10, 5] cover the entire observed range with margin,
with no effect on any real value.

**Flags retired:**
- `[COPULA-DEFECT-k_DD]` — resolved. Not a defect; mislabeled column.
- `[TANH-GAMMA-PROVISIONAL]` — resolved. Replaced by canonical form.

**Flags raised:**
- `[COPULA-COLUMN-IS-LOG-K-DD]` — documented in engine notes so downstream
  consumers know the copula column is log-scale.

## Fix 2 — synthetic-prior propagation

**Problem:** `work_g2/source_modes.csv` names ten parameters with
`resample_w_replacement` and `n_src=20000`. This reads as if 20,000
empirical observations exist. They do not. `notes.md` Layer 1 correctly
calls these "Synthetic priors (10)", but that designation was never
propagated into the engine, the runs, or the trace output.

**Fix:** `engine_v5.PARAM_PROVENANCE` maps every parameter to one of:
- `empirical_large_*` — O, C, E, A, N, DASS_*
- `empirical_medium_*` — S_M, S_P, S_N
- `empirical_small_*` — a_inhib, lambda_ineq, k_DD
- `empirical_tiny_*` — alpha_R, beta_choice, phi_per, w_MB (149 subjects)
- `SYNTHETIC_PRIOR` — the ten prior-only parameters
- `composite_of_*` — D, general_distress

`SYNTHETIC_PRIOR_PARAMS` = [theta_T, u_U, eta_vol, omega_social, lambda_norm,
s_spite, tau_sens, gamma_sal, kappa_ctx, m_t].

Every artifact now carries this list:
- `run_32k_v5_summary.txt` includes `n_synthetic_prior_params=10` and
  `synthetic_prior_params=...`.
- `xai_trace.trace_agent()` return dict includes `provenance`,
  `synthetic_priors_in_effect`, `active_flags`.

## Baseline shift

| Metric | P2i (pre-P4) | P4 (post-fix) | Delta |
|---|---|---|---|
| state_hash | ef1b5da9... | **2c238aca...** | (expected) |
| coop | 0.9625 | 0.9497 | −0.013 |
| gini | 0.4394 | 0.3960 | −0.043 |
| distress | 0.4640 | 0.4652 | ~0 |
| w_mb_eff | 0.4576 | 0.4569 | ~0 |
| defection | 33 | 60 | +27 |
| Q_net | 15.77 | 30.07 | ~2x |
| reward | 1.6479 | 2.1314 | +0.48 |
| mean_a2 | 2.0878 | 2.0635 | −0.024 |

The Gini drop and Q_net rise are consequences of the k_DD decode: agents
now have a real spread of discount factors, which changes long-horizon
learning dynamics. This is a scientifically meaningful shift, not a
numerical artifact.

## Trace re-verification

`test_xai_trace.py` re-run on new baseline:
- Reference hash: `2c238aca...`
- Trace agent 100: hash == reference. PASS
- Trace agent 0: hash == reference. PASS
- 9/9 checkpoint replays match.
- Trace return dict now includes the provenance block.

## Flags

### Retired
- `[COPULA-DEFECT-k_DD]` — resolved.
- `[TANH-GAMMA-PROVISIONAL]` — resolved.

### Raised
- `[COPULA-COLUMN-IS-LOG-K-DD]` — copula column is log-scale.
- `[SYNTHETIC-PRIOR-PARAMS]` — 10 params are prior-only, propagated to
  every artifact.

### Carried forward
- `[MF-GENERALIZED]`, `[REWARD-SCALE-ASYMMETRY]`,
  `[SYNTH-PRIOR-COEFF]`, `[INERT-DEFAULT]`.

## Remaining work (per previous audit)

1. **Fix #3** — the four two-step parameters have only 149 distinct values.
   Options: accept documented limitation, or refit hierarchically.
2. **Coupled-engine recovery** — never run. Must be run before any scientific
   claim.
3. **Human benchmark validation** — not started.
4. **Phase 5 — interventions and dashboard** — deferred until 1–3 are done.

## Out of scope (Phase 4)

No interventions. No dashboard. No refitting. Fixes only.
