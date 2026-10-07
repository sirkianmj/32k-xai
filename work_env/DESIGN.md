# work_env/DESIGN.md — Phase 1 environment

**Scope:** Minimal Fehr-Schmidt ultimatum environment on a Watts-Strogatz
small-world graph. Compute-feasibility analog of `work_g8`. Not the full
sandbox — see §8.

## 1. State (per agent, per tick)

| Field | Init | Role |
| :--- | :--- | :--- |
| `alpha` | Copula draw `[SYNTH-PRIOR]` | Envy |
| `beta` | Copula draw `[SYNTH-PRIOR]` | Guilt |
| `w_mb_base` | Copula draw | Arbitration weight |
| `distress` | `m_t` marginal | Dynamic state |
| `wealth` | 10.0 | Cumulative |
| `last_accept` | 1 | Prior interaction outcome |

Source: `synthetic_32k.pkl` from G2. **No Q-tables** — the ultimatum is a
one-shot decision per tick, not a learning task.

## 2. Network

Watts-Strogatz `(n=32_000, k=6, p=0.1)`. Adjacency stored as
`jnp.int32[N, 6]` neighbor table. **`[SYNTH-PRIOR]`** — not derived from
data. Swept as a sensitivity axis in Phase 6.

## 3. Utility (Fehr-Schmidt, N=2)

Proposer P, responder R, endowment E=10, offer o∈{0..10}. If R accepts:

    π_P = E − o          π_R = o
    U_P = π_P − α_P·max(π_R − π_P, 0) − β_P·max(π_P − π_R, 0)
    U_R = π_R − α_R·max(π_P − π_R, 0) − β_R·max(π_R − π_P, 0)

If R rejects: `U_P = U_R = 0`. **R accepts iff `U_R ≥ 0`** (deterministic
given α_R, β_R, o).

P's offer: `o* = argmax_o [ U_P(o) · 1[U_R(o) ≥ 0] ]`, evaluated over the
11 discrete offer levels. Full-information baseline (P observes R's
parameters). Bayesian-belief variant deferred to Phase 3.

## 4. Per-tick logic

1. **Match.** Each agent `i` picks one random neighbor `j = neighbors[i, k_i]`.
   A pair plays iff the choice is mutual (`j` also picked `i`). Non-mutual
   agents idle this tick.
2. **Role.** In each mutual pair, lower agent ID is proposer.
3. **Offer + accept/reject** per §3.
4. **Payoff.** Accept: `wealth_P += E − o`, `wealth_R += o`.
   Reject: no wealth change.
5. **State update.** `last_accept_P ← accept`;
   `distress_R ← 0.95·distress_R + 0.05·(1 − accept)`.
6. **w_mb** per §5.

**Activity rate.** With k=6, the probability that a random pick is mutual
is ≈ 1/6 ≈ 17%. This is a topological consequence of WS(k=6), not a bug.
See §9 for the alternative.

## 5. Distress → w_MB

`w_mb_eff = w_mb_base · (1 − distress · κ_decay)`, `κ_decay = 0.8`
**`[SYNTH-PRIOR]`**. Both `w_mb_base` and `w_mb_eff` are logged per agent
per tick.

## 6. Per-tick outputs

Written to `work_env/run_32k.log` as CSV:

    tick, cooperation_rate, wealth_gini, mean_distress,
    mean_w_mb_eff, defection_count

- `cooperation_rate` = accepted / (accepted + rejected)
- `wealth_gini` = Gini over the 32k wealth vector
- `defection_count` = rejected offers (proposer offered 0, or responder
  rejected a positive offer)

## 7. Determinism

Single seed `s`. `state_hash = SHA256(wealth_bytes || distress_bytes || tick)`,
logged at ticks {0, 100, 200, 300, 400, 500}. Two runs with the same `s`
must produce identical hashes. Different `s` → different hashes but the
same parameter draw (Copula is seeded separately).

## 8. Out of scope (Phase 1)

No orchestration, no XAI interrogation API, no dashboard, no exogenous
interventions, no learning (no Q-tables, no belief updates), no
second-order network effects. Phases 2–5.

## 9. Open questions (for review before engine.py)

1. **Partner selection.** Mutual-pick (current) gives ~17% activity rate
   per tick. Alternative: one-sided proposer-driven, half the agents act
   as proposers each tick and pick a neighbour; responders may play
   multiple times per tick. Which?
2. **Distress update form.** `0.95·d + 0.05·(1 − accept)` is provisional.
   Alternative: scale by relative wealth loss. Needs pinning down because
   it feeds the XAI trace schema.
3. **Full-information baseline.** P observes R's true α,β. Alternative:
   P uses the population-mean. Which for Phase 1?
4. **Parameter file.** Confirm `synthetic_32k.pkl` exists at the expected
   path with shape `(32000, d)`.
