# work_env — Phase 1 environment core — notes

**Commit target:** `G-ENV-P1` — environment engine core, 32k × 500 ticks.
**Date:** 07 October 2026.
**Hardware:** NVIDIA GTX 1660 Ti (6 GB VRAM, CC 7.5), WSL2 Ubuntu, JAX 0.11.2 CUDA 13.

## Deliverable

- `engine.py` — JAX-vectorized Fehr-Schmidt ultimatum on Watts-Strogatz.
- `run_32k.py` — driver: 500 ticks, per-tick CSV log, final state pickle, SHA-256 hash.
- `run_32k.log` — per-tick metrics.
- `final_state.pkl` — final wealth / distress / last_accept.
- `run_32k_summary.txt` — wall, hash, final aggregates.

## Result

- **Agents:** 32,000 on Watts-Strogatz (k=6, p=0.1), degree range [3, 10], mean 6.000.
- **Ticks:** 500.
- **Seed:** 42.
- **Wall time (GPU, post-compile):** 1.802 s.
- **First-call wall time (incl. compile):** ~4.4 s.
- **Final state hash:** `867d241fb058eb01f464387de84731c108d6d83ee95160b9f27f3b977849590c`.
- **Final cooperation rate:** 1.0000.
- **Final wealth Gini:** 0.0618.
- **Final mean distress:** 0.0575.
- **Final mean `w_mb_eff`:** 0.5924.
- **Final defection count:** 0.

## Findings

### Finding 1 — The distress → `w_mb_eff` link is a diagnostic, not a behavioural modulator

`w_mb_eff` is computed per tick and logged, but **it is not read by the ultimatum decision loop**. The Fehr-Schmidt offer/accept logic uses `alpha_R`, `beta_choice` (via population means for the proposer, true values for the responder). `w_mb_base` is a two-step arbitration weight that has no role in a one-shot ultimatum.

Consequence: the entire distress-modulation pathway specified in spec Part V.2 is **present as instrumentation only** in Phase 1. `w_mb_eff` will not change agent behavior until a task that actually consumes it (e.g., the two-step engine from G8, or a multi-tick decision task) is added. Flag: `[WMOD-NOT-WIRED]`.

This is intentional — Phase 1 is the minimal ultimatum. It is documented so that no downstream artifact mistakenly treats `w_mb_eff` as causal in the current engine.

### Finding 2 — 100% cooperation equilibrium under the population-mean rule

With `mu_alpha = 0.3661` and `mu_beta = 4.6474`, the proposer's optimal offer under the population-mean accept-constraint is the fair split (o = 5). All responders accept. Cooperation rate = 1.0, defection count = 0. Gini stays at 0.0618 — the residual inequality comes from the WS degree distribution (agents with more neighbors play more often) and from within-tick role asymmetries, not from strategic defection.

This is a **real behavioural prediction of the model as specified**, not a bug. It is also the scientifically correct default: the population-mean assumption is what spec §9.3 mandates, and the resulting equilibrium is exactly what a homogeneously fair population produces.

Phase 6 sensitivity sweeps should include: (a) heterogeneous beliefs (Bayesian per-pair estimates instead of population mean), (b) true-parameter observation, (c) tighter copula draws on `alpha_R` (higher envy → lower offers → more rejections).

### Finding 3 — First-run column-match bug (resolved)

First run selected `m_t` (state/mood, range [−3, 3]) instead of `general_distress` (DASS bifactor, range [0, 1]) due to a `_find_col` candidate-order issue. Corrected to prefer `general_distress`. Diagnostic signature of the bug: final mean distress was −0.0014, wrong-signed for a DASS-derived factor. Post-fix: 0.0575, in-range.

## Flagged synthetic priors

| Parameter | Value | Flag | Grounding |
| :--- | :--- | :--- | :--- |
| κ_decay | 0.8 | `[SYNTH-PRIOR]` | Not derived from data; spec Part V.2 form only |
| WS graph (n, k, p) | (32000, 6, 0.1) | `[SYNTH-PRIOR]` | Topology not fit to empirical network |
| Distress decay coeff | 0.95 | `[SYNTH-PRIOR]` | Provisional default |
| Distress rise coeff | 0.05 | `[SYNTH-PRIOR]` | Provisional default |
| Population-mean α, β | mu(α)=0.3661, mu(β)=4.6474 | `[POPULATION-MEAN-ASSIGNMENT]` | Proposers do not observe responder types |
| `w_mb_eff` wiring | not connected | `[WMOD-NOT-WIRED]` | Diagnostic only in Phase 1 |

All five must be swept as sensitivity axes in Phase 6.

## Parameter revisions (chronological)

| Time | Change | Reason |
| :--- | :--- | :--- |
| 2026-10-07 (initial) | κ_decay = 0.8, WS = (32000, 6, 0.1), distress = (0.95, 0.05) | Locked per §9 answers |
| 2026-10-07 (run 1) | distress column `m_t` → `general_distress` | Column-match bug, spec violation corrected |
| 2026-10-07 (run 2) | none | Post-fix canonical run |

No changes to κ_decay, WS, or distress coeffs after lock. No further revisions.

## Reproducibility

Rerun command:

    cd ~/projects/xai_project/work_env && python3 run_32k.py

Expected on the same hardware + seed: state_hash = `867d24...0c`. If a rerun produces a different hash, the compiler version, cuDNN, or copula file has changed; investigate before trusting any downstream result.

## Out of scope (Phase 1)

No checkpointing, no replay, no dashboard, no XAI interrogation API, no interventions, no orchestration. Those are Phase 2–5. The state store written here (`final_state.pkl`, `run_32k.log`) is the read-interface they will attach to.
