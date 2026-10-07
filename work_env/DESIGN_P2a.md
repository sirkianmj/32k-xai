# work_env/DESIGN_P2a.md — Dual-system social agent

**Scope:** Add Q-tables, dual-system arbitration, and w_mb_eff wiring to the
Phase 1 engine. Agents now learn across ticks. First engine whose behaviour
depends on w_mb (and therefore on distress and on the copula's w_MB draw).

**Not in scope:** XAI trace API, dashboard, interventions, sharding. Those
are Phase 2b / 3 / 4.

## Agent state per tick

Dynamic (evolves):
- Q_mf  : (2,)   TD values for stage-1 actions {engage=0, avoid=1}
- Q_td  : (2, 2) stage-2 values for (s2, a2); a2 ∈ {cooperate=0, defect=1}
- wealth, distress, last_action

Static (from synthetic_32k.pkl):
- alpha_R, beta_choice, w_MB, general_distress

## Per-tick decision (each agent independently)

1. **Model-based stage-1 values** (marginalize over transition):
   m[s2] = max_a2 Q_td[s2, a2]
   Q_mb[0] = P_COMMON·m[0] + (1−P_COMMON)·m[1]
   Q_mb[1] = P_COMMON·m[1] + (1−P_COMMON)·m[0]

2. **Arbitration:**
   w_mb_eff = clip(w_MB · (1 − distress · κ_decay), 0, 1)
   Q_net = w_mb_eff · Q_mb + (1 − w_mb_eff) · Q_mf

3. **Stage-1 choice** (softmax): a1 ~ Categorical(β · Q_net)
   a1 ∈ {0 = engage, 1 = avoid}

4. **Transition:** with P_COMMON = 0.7, s2 = a1; else s2 = 1 − a1

5. **Stage-2 choice** (softmax): a2 ~ Categorical(β · Q_td[s2, ·])
   a2 ∈ {0 = cooperate, 1 = defect}

## Social layer (interaction)

- WS mutual-pick selects pairs (same as Phase 1)
- Role: coin flip per pair → proposer / responder
- **Interaction happens iff both agents chose a1 = engage AND mutual pick**
- Proposer's offer: fair (o = 5) if a2 = 0, selfish (o = 2) if a2 = 1
- Responder accepts iff U_R(o) ≥ 0 (given responder's true α, β)
- If accept: proposer gains E − o, responder gains o. If reject: both 0.
- Non-engaged agents gain 0

## Learning

- r = agent's own payoff this tick (0 if not engaged)
- Q_td[s2, a2] += α · (r − Q_td[s2, a2])
- Q_mf[a1]    += α · (r − Q_mf[a1])

## Distress update

- Responders who rejected gain distress:
  distress ← 0.95 · distress + 0.05 · 1[rejected]
- Everyone else: distress ← 0.95 · distress

## Per-tick aggregates logged

tick, cooperation_rate, n_mutual_pairs, mean_distress, mean_w_mb_eff,
defection_count, wealth_gini, mean_abs_Q_net, mean_reward

## Flags (all Phase 6 sweep axes)

| Item | Flag | Grounding |
|---|---|---|
| κ_decay = 0.8 | [SYNTH-PRIOR] | Form from spec Part V.2 only |
| WS (32000, 6, 0.1) | [SYNTH-PRIOR] | Not fit to empirical network |
| Distress coeffs (0.95, 0.05) | [SYNTH-PRIOR] | Provisional |
| Offer levels (5, 2) | [SYNTH-PRIOR] | Phase 1 used 0..10 continuum; v2 uses binary |
| P_COMMON = 0.7 | [SYNTH-PRIOR] | Matches G8 benchmark |
| No sharding | [SCALE-1-SHARD] | Country-scale requires sharding (Phase 2b) |

## Design decisions from v1 changed in v2

1. **Population-mean offer rule removed.** v1 used mu(α), mu(β) to compute a
   fixed fair offer. v2 replaces this with Q-learning: each agent learns
   its stage-2 strategy. The [POPULATION-MEAN-ASSIGNMENT] flag from v1 is
   retired; [Q-LEARNED-OFFER] replaces it.

2. **Distress → w_mb_eff is now wired.** In v1 the metric was logged but
   unused. In v2 it is read by the stage-1 arbitration step, so it
   modulates behaviour. [WMOD-NOT-WIRED] retired.

3. **Reward is per-tick.** No discounting across ticks. This is
   intentional: the two-step task in G8 also has no γ, and adding one
   introduces a new free parameter requiring justification.
