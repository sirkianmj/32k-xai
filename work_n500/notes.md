# w_MB recovery (Daw two-step, N=500) — notes

**Source:** `Data_Xai32kProje/XAI_Markov_TwoStep_Data/data.csv`
(Zorowitz & Niv, 2021, Princeton University).
**Sample:** 149 subjects × 201 trials each.
**Task:** two-step Markov decision task (Daw et al., 2011 design).

## Model

Four-parameter hybrid two-step model (ts_par4):

- **α_R** — learning rate (single, shared across both stages)
- **β_choice** — inverse temperature (softmax)
- **φ_per** — perseveration (eligibility trace / repetition bonus)
- **w_MB** — model-based arbitration weight ∈ [0, 1]

Per-subject MLE via L-BFGS-B with 3 random restarts.

**Why 4 parameters and not 7:** the ts_par7 model (with asymmetric learning
rates α_R+/α_P− and stage-specific β) failed to converge on this data volume.
The 4-parameter reduction was necessary for identifiability. Flag:
`[COLLAPSED-LR]` on α_R, `[COLLAPSED-BETA]` on β_choice.

## Real-data fit (149 subjects)

| Parameter | Mean | SD | Median |
| :--- | ---: | ---: | ---: |
| α_R | 0.367 | 0.273 | 0.355 |
| β_choice | 4.650 | 3.576 | 3.916 |
| φ_per | 0.302 | 0.294 | 0.242 |
| w_MB | 0.623 | 0.331 | 0.693 |

## Parameter recovery (N=500 synthetic)

Same model, same per-subject MLE, 500 synthetic agents with known w_MB
drawn uniformly from [0, 1]:

| Metric | Value |
| :--- | ---: |
| Spearman ρ (w_MB) | **0.7824** |
| p-value | 1.65e−104 |
| bias | −0.0051 |
| RMSE | 0.2070 |
| Pass (ρ > 0.6) | YES |

## Gate status

- **G5 (partial)**: Daw paradigm recovery ρ = 0.7824 → PASS.
  Kool paradigm not yet executed → `[KOOL-VALIDATION PENDING]`.
- **G1 (parameter count)**: 4 of 21 parameters estimated from this dataset.

## Flags carried forward

- `[COLLAPSED-LR]` — α_R is shared positive/negative
- `[COLLAPSED-BETA]` — β is shared across stages
- `[KOOL-VALIDATION PENDING]` — Kool dataset not yet acquired
- `[RECOVERY-N500]` — full N=500 protocol satisfied

## Reproducibility

Script: `fit_and_recover.py`. Log: `fit_and_recover_n500.log`.
Runtime: 27 s (real fit) + 74 s (recovery). Run with:
cd work_n500 && python fit_and_recover.py

