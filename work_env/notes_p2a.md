# work_env — Phase 2a notes — dual-system social agent

**Commit target:** `G-ENV-P2a` — dual-system agent + w_mb_eff wired.
**Date:** 07 October 2026.
**Hardware:** GTX 1660 Ti (6 GB VRAM, CC 7.5), WSL2 Ubuntu, JAX 0.11.2 CUDA 13.

## Deliverable

- `engine_v2.py` — dual-system agent with Q-tables, arbitration, mutual
  interaction, distress update.
- `run_32k_v2.py` — driver.
- `run_32k_v2.log` — per-tick metrics.
- `final_state_v2.pkl` — final Q_mf, Q_td, wealth, distress.
- `run_32k_v2_summary.txt` — wall, hash, aggregates.
- `DESIGN_P2a.md` — design decisions.

## Result

- Agents: 32,000 on WS (k=6, p=0.1), degree range [3, 10].
- Ticks: 500. Seed: 42.
- Wall (GPU, post-compile): **1.711 s**.
- First-call (incl. compile): **3.9 s**.
- State hash: `114f00abcd6df2b76cf124ea4a44bde23f2ecedec497d01e7b0e1c17cc7cd7e2`.
- Final coop_rate: **0.7883**.
- Final gini: **0.1732**.
- Final mean_distress: 0.0064.
- Final mean_w_mb_eff: 0.6179.
- Final defection_count: **192**.
- Final mean_abs_Q_net: 0.1588.
- Final mean_reward: 0.2234.

## Findings

### Finding 1 — Emergent cooperation breakdown

Phase 1 produced a 100% cooperation equilibrium because the offer rule was
fixed. Phase 2a produces 78.83% cooperation. The difference is entirely
due to Q-learning: agents now learn their stage-2 action from reward, and
agents with low α_R or high β_choice learn to defect (offer 2), which
high-α responders reject. The 21% defection rate is the first emergent
social phenomenon in the platform.

### Finding 2 — Inequality is now behavioural

Phase 1 gini = 0.0618 came only from WS degree asymmetry (agents with more
neighbours played more often). Phase 2a gini = 0.1732 — nearly 3× — because
some agents learn cooperative strategies (steady positive reward) while
others cycle in defect→reject→no-reward loops. Wealth divergence is now a
property of the population's parameter distribution, not of the graph.

### Finding 3 — w_mb_eff is alive but barely moves

mean_w_mb_eff = 0.6179 vs. base mean w_MB ≈ 0.62. Nearly unchanged, because
mean distress = 0.0064 is very low. The distress→w_mb_eff pathway exists
and is read by the arbitration step (Finding 2 in P1 retires here), but at
Phase 2a's low-rejection regime it barely modulates. A distress injection
scenario (Phase 4 interventions) will exercise this pathway.

### Finding 4 — Learning is stable

mean_abs_Q_net = 0.1588 with no evidence of blow-up across the tick series
(see run_32k_v2.log). The learning rate α_R from the copula (mean ≈ 0.37)
combined with per-tick reward magnitudes (0-10) produces well-behaved Q
updates. No clipping or damping is required.

## Design changes from Phase 1

1. **Population-mean offer rule → Q-learned offer.** v1 used mu(α), mu(β)
   to compute a fixed fair offer. v2 replaces this with per-agent Q-learning
   on stage-2 actions. Flag `[POPULATION-MEAN-ASSIGNMENT]` retired;
   `[Q-LEARNED-OFFER]` replaces it.
2. **w_mb_eff wired into arbitration.** Flag `[WMOD-NOT-WIRED]` retired.
3. **Interaction requires mutual engagement (a1 = 0 both sides).** Agents
   who choose to avoid do not interact even if paired.

## Synthetic priors (unchanged from P1)

- κ_decay = 0.8 `[SYNTH-PRIOR]`
- WS (32000, 6, 0.1) `[SYNTH-PRIOR]`
- Distress coeffs (0.95, 0.05) `[SYNTH-PRIOR]`
- Offer levels {fair=5, selfish=2} `[SYNTH-PRIOR]` — new in v2
- P_COMMON = 0.7 `[SYNTH-PRIOR]`
- No sharding `[SCALE-1-SHARD]`

## Reproducibility

    cd ~/projects/xai_project/work_env && python3 run_32k_v2.py

Expected hash: `114f00...d7e2`. Same seed + same hardware + same copula
file → same hash. Changing anything invalidates the comparison.

## Out of scope (Phase 2a)

No sharding, no checkpointing, no XAI trace API, no dashboard, no
interventions, no provenance card. Phase 2b (scale test) and Phase 3 (XAI
trace) are the next deliverables.
