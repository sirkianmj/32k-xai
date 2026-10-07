# work_env — Phase 2h notes — coefficient sensitivity sweep

**Commit:** `G-ENV-P2h` — sensitivity ranking of 14 coupling constants.
**Date:** 07 October 2026.
**Hardware:** GTX 1660 Ti (6 GB VRAM), WSL2 Ubuntu, JAX 0.11.2 CUDA 13.

## Deliverable

- `engine_v5_sweep.py` — per-axis {x0.5, x1.0, x2.0} sensitivity screen at
  N=10k, then full-resolution rerun of top-3 at N=32k.
- `sweep_p2h.json`, `sweep_p2h.log` — full results.

## Sensitivity ranking (score = |dCoop| + |dGini| + |dA2|)

| Rank | Axis | dCoop | dGini | dA2 | score |
|---|---|---|---|---|---|
| 1 | TAU_BASE | 0.0207 | 0.1609 | 0.0084 | 0.1900 |
| 2 | SM_DEFECT | 0.0017 | 0.0075 | 0.1546 | 0.1638 |
| 3 | T_THREAT | 0.0182 | 0.1164 | 0.0062 | 0.1408 |
| 4 | TRUST_K | 0.0118 | 0.0216 | 0.0946 | 0.1280 |
| 5 | E_ENGAGE | 0.0189 | 0.0743 | 0.0063 | 0.0996 |
| 6 | U_UNCERT | 0.0118 | 0.0792 | 0.0039 | 0.0949 |
| 7 | SN_STATUS | 0.0004 | 0.0033 | 0.0828 | 0.0865 |
| 8 | LAMBDA_NORM_A | 0.0235 | 0.0014 | 0.0009 | 0.0259 |
| 9 | A_INHIB | 0.0032 | 0.0125 | 0.0015 | 0.0171 |
| 10 | KAPPA_DECAY | 0.0007 | 0.0089 | 0.0066 | 0.0162 |
| 11 | C_BOOST | 0.0009 | 0.0054 | 0.0019 | 0.0082 |
| 12 | SAL_GAIN | 0.0009 | 0.0021 | 0.0052 | 0.0082 |
| 13 | N_DISTRESS | 0.0000 | 0.0001 | 0.0001 | 0.0002 |
| 14 | SP_PUNISH | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Findings

### Finding A — SP_PUNISH is dead code (BUG)

The score is exactly 0.0000. Root cause in engine_v5.py, the stage-2
logits line:

    dark_bias = -SM_b - SN_b + SP_b * 0.0

The trailing multiplication by 0.0 zeroes the SP_b contribution. The
parameter is read (the audit passes) but contributes nothing to
behaviour. This is a mistake, not a design choice. Must be fixed
before SP_PUNISH can be declared inert or load-bearing.

### Finding B — TAU_BASE dominates wealth inequality

Sweeping TAU_BASE {1.0, 2.0, 4.0} at N=32k drives Gini {0.4023, 0.5627,
0.7118}. Exploration temperature is the single largest driver of wealth
inequality in this model. dGini = 0.31 across the sweep range. This is a
scientifically important finding worth a dedicated experiment.

### Finding C — Four axes are effectively inert

- SP_PUNISH (dead code, see A)
- N_DISTRESS (score 0.0002)
- C_BOOST, SAL_GAIN (score 0.0082 each)

These four can be left at base value in Phase 6 empirical fitting
without materially changing the qualitative picture. Each carries
[INERT-DEFAULT] going forward.

### Finding D — Two axes families operate on different macro variables

- TAU_BASE and T_THREAT primarily move Gini (via engagement patterns).
- SM_DEFECT and TRUST_K primarily move mean_a2 (via action selection).

This means Phase 6 empirical fitting can decompose: TAU/T_THREAT should
be calibrated against Gini-curve data, while SM/SN/TRUST_K should be
calibrated against strategy-distribution data (mean_a2 or its analogue).

## Full-resolution top-3 (N=32k)

| Axis | x0.5 | x1.0 | x2.0 |
|---|---|---|---|
| TAU_BASE | coop=0.9731 gini=0.4023 | coop=0.9611 gini=0.5627 | coop=0.9571 gini=0.7118 |
| SM_DEFECT | coop=0.9241 gini=0.5666 a2=1.9741 | coop=0.9611 gini=0.5627 a2=2.1110 | coop=0.9826 gini=0.5562 a2=2.2677 |
| T_THREAT | coop=0.9740 gini=0.4468 | coop=0.9611 gini=0.5627 | coop=0.9286 gini=0.6292 |

## Flags raised

### [INERT-DEFAULT] (new)

SP_PUNISH, N_DISTRESS, C_BOOST, SAL_GAIN carry this flag. Phase 6 may
skip empirical fitting for these axes if the inertness holds after the
SP_PUNISH bug is fixed.

### [SP-PUNISH-DEAD-CODE] (new)

engine_v5's SP_b contribution is multiplied by 0.0. Must be corrected in
Phase 2i or the sweep must be re-run with the fix applied. Until then,
SP_PUNISH's inertness is an artifact.

### Carried forward

[NONINTERACTING-Q-DRIFT], [REWARD-SCALE-ASYMMETRY], [COPULA-DEFECT-k_DD],
[TANH-GAMMA-PROVISIONAL], [MF-GENERALIZED], [SYNTH-PRIOR-COEFF].

## Out of scope (Phase 2h)

No empirical fitting. No coefficient values updated. Sweep is diagnostic
only. The empirical correction is Phase 6 (post-XAI-trace).

## Cost

- Total wall: 3m17s (14 axes x 3 values x 2.5s at N=10k + 3 axes x 3 x 2.0s at N=32k)
- No failures, no OOMs.
