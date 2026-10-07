# work_env — Phase 2g notes — deterministic checkpoints + replay

**Commit:** `G-ENV-P2g` — deterministic replay from any checkpoint.
**Date:** 07 October 2026.
**Hardware:** GTX 1660 Ti (6 GB VRAM), WSL2 Ubuntu, JAX 0.11.2 CUDA 13.

## Deliverable

- `engine_v5_ckpt.py` — `run_with_checkpoints()` and `replay_from_checkpoint()`.
- `test_replay.py` — bit-for-bit replay verification.
- `ckpt_p2g/tick_XXXX.npz` — 10 checkpoints at ticks 50..500.

## Result

Reference run (no checkpoints): 500 ticks, wall 1.83 s, hash `302b7a8796419405cb21116d0f3949f9dc9a32f5667305dc026ebd6a147cd1bb`.

This hash matches the **committed Phase 2e run** exactly. Cross-phase
reproducibility confirmed.

Checkpoint run (every 50 ticks): 10 checkpoints, wall 1.83 s, final hash
identical to reference.

Replay verification table:

| Replay from | Target | Hash | Result |
|---|---|---|---|
| t=50  | 500 | 302b7a... | PASS |
| t=100 | 500 | 302b7a... | PASS |
| t=150 | 500 | 302b7a... | PASS |
| t=200 | 500 | 302b7a... | PASS |
| t=250 | 500 | 302b7a... | PASS |
| t=300 | 500 | 302b7a... | PASS |
| t=350 | 500 | 302b7a... | PASS |
| t=400 | 500 | 302b7a... | PASS |
| t=450 | 500 | 302b7a... | PASS |

**9/9 replay hashes match. Bit-for-bit deterministic. PASS.**

## What this satisfies

Spec Part VI.2 — deterministic replay architecture:

> "Only the macro-state and RNG keys are logged at each tick. When a query
> is initiated at tick t, the API re-simulates the target agent's specific
> trajectory from the nearest checkpoint on the CPU to extract Q_MB, Q_MF,
> and τ_i."

Checkpoints now exist. Replay is verified. XAI trace Phase 3 has a working
substrate to build on.

## Implementation notes

- Checkpoints stored as `.npz` (numpy compressed), not pickle. Portable,
  no code execution on load, no version-drift surface.
- Each checkpoint carries: Q_mf, Q_s2, P_hat, wealth, distress, last_a1,
  tick, seed, n_ticks, N, state_hash.
- The RNG is not saved explicitly. Replay splits the seed fresh and slices
  to the target range. This works because `random.split(PRNGKey(seed), n_ticks)`
  is deterministic and the scan consumes keys in order. Verified by 9/9
  replay matches.
- Memory: each 32k-agent checkpoint is ~9 MB on disk (compressed).
  At `ckpt_every=50` over 500 ticks that is ~90 MB per run. Acceptable.

## Flags carried forward

- `[NONINTERACTING-Q-DRIFT]`, `[REWARD-SCALE-ASYMMETRY]`, `[COPULA-DEFECT-k_DD]`,
  `[TANH-GAMMA-PROVISIONAL]`, `[MF-GENERALIZED]`, `[SYNTH-PRIOR-COEFF]`.
- No new flags.

## Out of scope (Phase 2g)

No XAI trace API yet (Phase 3). No coefficient sweeps (Phase 2h). No
dashboard (Phase 4+). Replay works at the carry level; per-agent
interrogation is Phase 3.
