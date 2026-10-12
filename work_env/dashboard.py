# dashboard.py — Phase 5: FastAPI backend for the 32k-XAI dashboard.
#
# Design-oriented endpoints for experimental use:
#   GET  /api/traits                    — list copula columns with stats
#   POST /api/design                    — full experiment: population mods +
#                                          interventions + batch seeds
#   POST /api/run/{name}                — pre-built scenario (legacy)
#   POST /api/run-custom                — ad-hoc interventions (legacy)
#   GET  /api/trace/{name}/{agent_i}    — single-agent decision trace
#   GET  /                               — dashboard_ui.html
import os, sys, json, uuid, shutil, time
from pathlib import Path
import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
os.chdir(WORK)

DEFAULT_COPULA = Path.home() / "projects/xai_project/work_g2/synthetic_32k_p6d.pkl"
os.environ.setdefault("P6C_PARAMS", str(DEFAULT_COPULA))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response

from scenarios import SCENARIOS
from xai_trace_scenarios import trace_in_scenario, shock_state_at_tick

app = FastAPI(title="32k-XAI Dashboard", version="0.3.0")

BENCH = ["UG mean offer", "UG rejection", "TG mean send", "TG mean return", "Final Gini"]
TARGET = [0.400, 0.160, 0.502, 0.372, 0.400]
SE = [0.050, 0.050, 0.130, 0.120, 0.120]

# Friendly labels for copula columns; falls back to raw name if missing.
TRAIT_LABELS = {
    "O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
    "A": "Agreeableness", "N": "Neuroticism",
    "S_M": "Machiavellianism", "S_P": "Psychopathy", "S_N": "Narcissism",
    "D": "Dark Core (general)",
    "DASS_Dep": "Depression", "DASS_Anx": "Anxiety", "DASS_Str": "Stress",
    "general_distress": "General distress",
    "alpha_R": "Learning rate", "beta_choice": "Decision sharpness",
    "phi_per": "Perseveration", "w_MB": "Model-based weight",
    "k_DD": "Delay discounting (log)", "a_inhib": "Inhibitory control",
    "lambda_ineq": "Inequity aversion", "theta_T": "Threat sensitivity",
    "u_U": "Uncertainty aversion", "eta_vol": "Volatility learning",
    "omega_social": "Social value weight", "gamma_sal": "Salience gain",
    "m_t": "Current mood",
}

# ---- Provenance record for every trait the dashboard knows about.
# status meanings:
#   GROUNDED   -- column came from a direct empirical sample (HCP, IPIP, PANAS, ...)
#   PROXY      -- column is a defensible proxy for the target construct;
#                 documented as weaker evidence than direct grounding
#   DERIVED    -- column is a deterministic function of other columns
#                 (e.g. bifactor general factor)
#   SYNTHETIC  -- literature-derived prior; flagged [SYNTHETIC-PRIOR-PARAMS]
#                 in every XAI trace
#   INERT      -- retired: present in the engine's parameter list but never
#                 read by the calibrated decision loop
TRAIT_PROVENANCE = {
    # --- Big Five: IPIP-FFM, N = 603k ---
    "O":  {"status": "GROUNDED", "source": "IPIP-FFM (N=603k)",
           "n": 603000, "note": "Openness to Experience, IPIP-FFM normative sample."},
    "C":  {"status": "GROUNDED", "source": "IPIP-FFM (N=603k)",
           "n": 603000, "note": "Conscientiousness, IPIP-FFM normative sample."},
    "E":  {"status": "GROUNDED", "source": "IPIP-FFM (N=603k)",
           "n": 603000, "note": "Extraversion, IPIP-FFM normative sample."},
    "A":  {"status": "GROUNDED", "source": "IPIP-FFM (N=603k)",
           "n": 603000, "note": "Agreeableness, IPIP-FFM normative sample."},
    "N":  {"status": "GROUNDED", "source": "IPIP-FFM (N=603k)",
           "n": 603000, "note": "Neuroticism, IPIP-FFM normative sample."},
    # --- Dark Triad bifactor: SD3, N = 17k ---
    "S_M": {"status": "GROUNDED", "source": "SD3 (N=17k)",
            "n": 17000, "note": "Machiavellianism specific factor."},
    "S_P": {"status": "GROUNDED", "source": "SD3 (N=17k)",
            "n": 17000, "note": "Psychopathy specific factor."},
    "S_N": {"status": "GROUNDED", "source": "SD3 (N=17k)",
            "n": 17000, "note": "Narcissism specific factor."},
    "D":   {"status": "DERIVED",  "source": "derived from S_M, S_P, S_N",
            "n": None,  "note": "General Dark Core, first bifactor component."},
    # --- DASS-21 bifactor: N = 40k ---
    "DASS_Dep": {"status": "GROUNDED", "source": "DASS-21 (N=40k)",
                 "n": 40000, "note": "Depression subscale."},
    "DASS_Anx": {"status": "GROUNDED", "source": "DASS-21 (N=40k)",
                 "n": 40000, "note": "Anxiety subscale."},
    "DASS_Str": {"status": "GROUNDED", "source": "DASS-21 (N=40k)",
                 "n": 40000, "note": "Stress subscale."},
    "general_distress": {"status": "DERIVED",
                         "source": "bifactor general factor",
                         "n": None, "note": "General distress factor."},
    # --- Learning / decision ---
    "alpha_R":     {"status": "GROUNDED", "source": "Daw two-step (N=149)",
                    "n": 149, "note": "Collapsed learning rate. [COLLAPSED-LR]."},
    "beta_choice": {"status": "GROUNDED", "source": "Daw two-step (N=149)",
                    "n": 149, "note": "Inverse temperature."},
    "phi_per":     {"status": "GROUNDED", "source": "Daw two-step (N=149)",
                    "n": 149, "note": "Perseveration (stage-1 stickiness)."},
    "w_MB":        {"status": "GROUNDED", "source": "Daw (N=149) + Kool recovery",
                    "n": 149, "note": "Model-based arbitration weight."},
    "k_DD":        {"status": "GROUNDED", "source": "delay-discounting (N=476)",
                    "n": 476, "note": "Log hyperbolic discount rate."},
    "a_inhib":     {"status": "GROUNDED", "source": "SSRT (N=264)",
                    "n": 264, "note": "Inhibitory control index."},
    "lambda_ineq": {"status": "GROUNDED", "source": "ultimatum (N=111)",
                    "n": 111, "note": "Fehr-Schmidt alpha. [ENVY-ONLY] simplification."},
    # --- P6d external proxies ---
    "theta_T":     {"status": "PROXY", "source": "HCP EMOTION FACE_MEDIAN_RT (N=482)",
                    "n": 482, "note": "Threat sensitivity, RT-based proxy."},
    "u_U":         {"status": "PROXY", "source": "HCP GAMBLING PUNISH_PROP_LARGER (N=491)",
                    "n": 491, "note": "Uncertainty aversion, choice-based proxy."},
    "omega_social":{"status": "PROXY", "source": "trust-game investment (N=113)",
                    "n": 113, "note": "Social value weight, proxy from behaviour."},
    "m_t":         {"status": "GROUNDED", "source": "PANAS (N=329)",
                    "n": 329, "note": "Current mood, direct ground."},
    # --- Synthetic ---
    "eta_vol":     {"status": "SYNTHETIC", "source": "literature prior",
                    "n": None, "note": "Volatility update. [SYNTHETIC-PRIOR-PARAMS]."},
    "gamma_sal":   {"status": "SYNTHETIC", "source": "literature prior",
                    "n": None, "note": "Salience gain. [SYNTHETIC-PRIOR-PARAMS]."},
    # --- Retired inert columns (kept for provenance completeness) ---
    "lambda_norm": {"status": "INERT", "source": "retired P6d step 1",
                    "n": None, "note": "Not read by the calibrated engine."},
    "s_spite":     {"status": "INERT", "source": "retired P6d step 1",
                    "n": None, "note": "Not read by the calibrated engine."},
    "tau_sens":    {"status": "INERT", "source": "retired P6d step 1",
                    "n": None, "note": "Not read by the calibrated engine."},
    "kappa_ctx":   {"status": "INERT", "source": "retired P6d step 1",
                    "n": None, "note": "Not read by the calibrated engine."},
}

def _load_p6c():
    import importlib.util
    p6c_path = WORK / "p6c_v5.py"
    spec = importlib.util.spec_from_file_location("p6c_dash", p6c_path)
    p6c = importlib.util.module_from_spec(spec)
    sys.modules["p6c_dash"] = p6c
    spec.loader.exec_module(p6c)
    return p6c

def _load_copula():
    import pandas as pd
    return pd.read_pickle(os.environ.get("P6C_PARAMS", str(DEFAULT_COPULA)))

def _apply_copula_ops(df, ops):
    """Return a modified copy of df. Each op is a dict:
      {"op": "shift_mean", "trait": "N", "value": 0.2}
      {"op": "scale_sd",   "trait": "N", "value": 1.5}
      {"op": "set",        "trait": "N", "value": 0.5}
      {"op": "clip_pct",   "trait": "N", "lo": 0.05, "hi": 0.95}
    """
    df = df.copy()
    for op in ops or []:
        trait = op.get("trait")
        if trait not in df.columns:
            raise ValueError(f"unknown trait '{trait}'")
        kind = op.get("op")
        col = df[trait].to_numpy().astype(np.float64)
        if kind == "shift_mean":
            col = col + float(op["value"])
        elif kind == "scale_sd":
            mu = col.mean()
            col = mu + (col - mu) * float(op["value"])
        elif kind == "set":
            col = np.full_like(col, float(op["value"]))
        elif kind == "clip_pct":
            lo = np.quantile(col, float(op["lo"]))
            hi = np.quantile(col, float(op["hi"]))
            col = np.clip(col, lo, hi)
        else:
            raise ValueError(f"unknown op '{kind}'")
        df[trait] = col.astype(np.float32)
    return df

def _resolve_target_mask(df, spec):
    """Convert a shock spec with target_where into a bool array of length N."""
    if "target_where" not in spec or spec["target_where"] is None:
        return None
    w = spec["target_where"]
    trait = w["trait"]
    if trait not in df.columns:
        raise ValueError(f"unknown trait '{trait}' in target_where")
    col = df[trait].to_numpy().astype(np.float64)
    q = w.get("quantile")
    if q is not None:
        threshold = np.quantile(col, float(q))
    else:
        threshold = float(w["value"])
    op = w.get("op", ">=")
    if   op == ">=": mask = col >= threshold
    elif op == ">":  mask = col >  threshold
    elif op == "<=": mask = col <= threshold
    elif op == "<":  mask = col <  threshold
    else: raise ValueError(f"unknown op '{op}' in target_where")
    return mask

def _benchmarks_block(plat):
    rows = []
    for name, p, t, se in zip(BENCH, plat, TARGET, SE):
        rows.append({
            "name": name,
            "platform": float(p),
            "target": float(t),
            "se": float(se),
            "delta": float(p - t),
            "pass": bool(abs(p - t) < se),
        })
    return rows

def _run_once(copula_path, shocks, N, T, seed):
    os.environ["P6C_PARAMS"] = str(copula_path)
    p6c = _load_p6c()
    if str(p6c.PARAMS) != str(copula_path):
        raise RuntimeError(f"copula override failed: {p6c.PARAMS} != {copula_path}")
    import jax.numpy as jnp
    theta = jnp.array([json.load(open(WORK / "p6c_results.json"))["calib"][k]
                       for k in p6c.CALIB_NAMES], jnp.float32)
    t0 = time.time()
    _, SIM = p6c.build_simulator(N, T, shocks=shocks)
    _ = SIM(theta, seed_val=int(seed)).block_until_ready()
    plat = np.asarray(SIM(theta, seed_val=int(seed)))
    return plat, time.time() - t0

def _run_design(copula_ops, shocks, N, T, seeds):
    """Run across seeds; return mean, per-seed, timeseries, and subgroups (from seed 0)."""
    df = _load_copula()
    if copula_ops:
        df = _apply_copula_ops(df, copula_ops)
    rng = np.random.default_rng(int(seeds[0]))
    idx = rng.integers(0, len(df), size=int(N))
    df = df.iloc[idx].reset_index(drop=True)
    resolved_shocks = []
    treated_mask = np.zeros(N, dtype=bool)
    for s in (shocks or []):
        s2 = dict(s)
        mask = _resolve_target_mask(df, s)
        if mask is not None:
            s2["target_mask"] = mask
            s2.pop("target_where", None)
            treated_mask |= mask
        resolved_shocks.append(s2)
    # Register this exact agent matrix under a run_id so the 3D views
    # can show the actual agents the engine will see.
    run_id = _run_id_for(copula_ops, N, seeds[0], shocks=resolved_shocks)
    _store_run(run_id, df)

    tmp = Path("/tmp") / f"p6c_design_{uuid.uuid4().hex}.pkl"
    df.to_pickle(tmp)
    prev_copula = os.environ.get("P6C_PARAMS", str(DEFAULT_COPULA))
    try:
        per_seed = []; walls = []
        cap0 = wealth0 = None
        for i, seed in enumerate(seeds):
            if i == 0:
                plat, cap, wealth = _run_once_with_diag(tmp, resolved_shocks, N, T, seed)
                cap0 = cap; wealth0 = wealth
            else:
                plat, w = _run_once(tmp, resolved_shocks, N, T, seed)
            per_seed.append(plat)
            walls.append(w if i > 0 else 1.0)
        per_seed_arr = np.array(per_seed)
        mean = per_seed_arr.mean(axis=0)
        std = per_seed_arr.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(5)

        # Time series and subgroups from seed 0
        ts_full = _timeseries_of(cap0)
        shock_ticks = _build_shock_tick_mask(resolved_shocks, T)
        ts_shock = _timeseries_of(cap0, shock_ticks=shock_ticks) if shock_ticks.any() else None

        subgroups = None
        if treated_mask.any() and not treated_mask.all():
            t_full  = _compute_final_metrics(cap0[:, treated_mask, :],  wealth0[treated_mask])
            c_full  = _compute_final_metrics(cap0[:, ~treated_mask, :], wealth0[~treated_mask])
            if shock_ticks.any():
                t_shock = _compute_final_metrics(cap0[shock_ticks][:, treated_mask, :],  wealth0[treated_mask])
                c_shock = _compute_final_metrics(cap0[shock_ticks][:, ~treated_mask, :], wealth0[~treated_mask])
            else:
                t_shock = c_shock = None
            effect_full  = {k: t_full[k]  - c_full[k]  for k in t_full}
            effect_shock = ({k: t_shock[k] - c_shock[k] for k in t_shock}
                            if t_shock is not None else None)
            subgroups = {
                "n_treated": int(treated_mask.sum()),
                "n_control": int((~treated_mask).sum()),
                "treated_full":  t_full,  "control_full":  c_full,  "effect_full":  effect_full,
                "treated_shock": t_shock, "control_shock": c_shock, "effect_shock": effect_shock,
            }

        # Enrich the cached df with final per-agent outcomes from seed 0.
        # The cap tensor in this engine has 10 columns (game-outcome
        # indicators), not 13, so per-agent distress and w_mb are not
        # available from it. We expose final wealth, which is what the
        # engine actually computes per agent.
        try:
            import pandas as _pd
            df_rich = df.copy()
            if wealth0 is not None:
                w = np.asarray(wealth0, dtype=np.float64)
                df_rich["final_wealth"] = w
                df_rich["final_wealth_log1p"] = np.log1p(np.maximum(w, 0.0))
                # simple economic outcome: rank within the population
                r = np.argsort(np.argsort(w))
                df_rich["final_wealth_rank_pct"] = r / max(1, len(w) - 1)
            _store_run(run_id, df_rich)
            print(f"[run_id={run_id}] enriched df with final_wealth columns")
        except Exception as _e:
            print(f"[run_id={run_id}] enrichment failed: {_e}")

        return {
            "mean": mean.tolist(),
            "std_over_seeds": std.tolist(),
            "per_seed": per_seed_arr.tolist(),
            "n_seeds": len(seeds),
            "wall_total_s": float(sum(walls)),
            "timeseries_full": ts_full,
            "timeseries_shock": ts_shock,
            "subgroups": subgroups,
            "run_id": run_id,
            "n_agents": int(N),
        }
    finally:
        os.environ["P6C_PARAMS"] = prev_copula
        try: tmp.unlink()
        except Exception: pass


def _compute_timeseries(cap_np):
    """Per-tick aggregate benchmarks from a (T, N, 10) cap tensor."""
    T = cap_np.shape[0]
    c0 = cap_np[:,:,0]; c1 = cap_np[:,:,1]; c2 = cap_np[:,:,2]
    c4 = cap_np[:,:,4]; c5 = cap_np[:,:,5]; c7 = cap_np[:,:,7]; c8 = cap_np[:,:,8]; c9 = cap_np[:,:,9]

    m_off = c0 > 0; n_off = m_off.sum(axis=1)
    offer_t = np.where(n_off > 0, (c0 * m_off).sum(axis=1) / np.maximum(n_off, 1) / 10.0, 0.0)

    m_resp = c2 > 0; n_resp = m_resp.sum(axis=1)
    rej_t = np.where(n_resp > 0, 1.0 - (c1 * m_resp).sum(axis=1) / np.maximum(n_resp, 1), 0.5)

    m_send = c5 > 0; n_send = m_send.sum(axis=1)
    send_t = np.where(n_send > 0, (c4 * m_send).sum(axis=1) / np.maximum(n_send, 1) / 10.0, 0.0)

    m_ret = (c9 > 0) & (c7 > 0); n_ret = m_ret.sum(axis=1)
    ret_frac_t = np.where(c7 > 0, c8 / np.maximum(c7, 1), 0.0)
    ret_t = np.where(n_ret > 0, (ret_frac_t * m_ret).sum(axis=1) / np.maximum(n_ret, 1) / 3.0, 0.0)

    return {
        "ticks": list(range(T)),
        "ug_offer": offer_t.tolist(),
        "ug_rej":   rej_t.tolist(),
        "tg_send":  send_t.tolist(),
        "tg_ret":   ret_t.tolist(),
    }


def _gini(wealth):
    w = np.asarray(wealth, dtype=np.float64)
    if len(w) < 2: return 0.0
    ws = np.sort(w); n = len(ws)
    cum = np.cumsum(ws); tot = cum[-1]
    return float((n + 1 - 2.0 * cum.sum() / tot) / n) if tot > 0 else 0.0


def _compute_final_metrics(cap_sub, wealth_sub):
    """5 benchmarks from caps restricted to a subgroup."""
    if cap_sub.shape[1] == 0:
        return {"ug_offer": 0.0, "ug_rej": 0.0, "tg_send": 0.0, "tg_ret": 0.0, "gini": 0.0}
    c0 = cap_sub[:,:,0]; c1 = cap_sub[:,:,1]; c2 = cap_sub[:,:,2]
    c4 = cap_sub[:,:,4]; c5 = cap_sub[:,:,5]; c7 = cap_sub[:,:,7]; c8 = cap_sub[:,:,8]; c9 = cap_sub[:,:,9]
    om = c0.reshape(-1); omsk = om > 0
    ug_offer = float((om * omsk).sum() / max(omsk.sum(), 1) / 10.0) if omsk.any() else 0.0
    rm = c2.reshape(-1) > 0
    ug_rej = float(1.0 - (c1.reshape(-1) * rm).sum() / max(rm.sum(), 1)) if rm.any() else 0.5
    sm = c5.reshape(-1) > 0
    tg_send = float((c4.reshape(-1) * sm).sum() / max(sm.sum(), 1) / 10.0) if sm.any() else 0.0
    tm = c9.reshape(-1) > 0; recv = c7.reshape(-1); ret = c8.reshape(-1)
    valid = tm & (recv > 0)
    tg_ret = float(((ret * valid) / np.where(recv > 0, recv, 1)).sum() / max(valid.sum(), 1) / 3.0) if valid.any() else 0.0
    return {"ug_offer": ug_offer, "ug_rej": ug_rej, "tg_send": tg_send, "tg_ret": tg_ret, "gini": _gini(wealth_sub)}


def _run_once_with_diag(copula_path, shocks, N, T, seed):
    os.environ["P6C_PARAMS"] = str(copula_path)
    p6c = _load_p6c()
    if str(p6c.PARAMS) != str(copula_path):
        raise RuntimeError("copula override failed")
    import jax.numpy as jnp
    theta = jnp.array([json.load(open(WORK / "p6c_results.json"))["calib"][k]
                       for k in p6c.CALIB_NAMES], jnp.float32)
    _, SIM = p6c.build_simulator(N, T, shocks=shocks, return_diag=True)
    m, cap, wealth = SIM(theta, seed_val=int(seed))
    return np.asarray(m), np.asarray(cap), np.asarray(wealth)


def _timeseries_of(cap_np, mask=None, shock_ticks=None):
    """Per-tick aggregate restricted to a subgroup and/or a tick mask."""
    if mask is not None:
        cap_np = cap_np[:, mask, :]
    if shock_ticks is not None:
        cap_np = cap_np[shock_ticks, :, :]
    return _compute_timeseries(cap_np)


def _build_shock_tick_mask(shocks, T):
    m = np.zeros(T, dtype=bool)
    for s in (shocks or []):
        t0 = int(s["tick"]); t1 = min(T, t0 + int(s["duration"]))
        if t1 > t0: m[t0:t1] = True
    return m


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.3.0"}

@app.get("/api/scenarios")
def list_scenarios():
    return {name: spec["description"] for name, spec in SCENARIOS.items()}

@app.get("/api/traits")
def list_traits():
    """Per-column descriptive statistics for the copula, plus a
    provenance record for every trait the dashboard recognises."""
    df = _load_copula()
    out = []
    for c in df.columns:
        v = df[c].to_numpy().astype(np.float64)
        q = np.percentile(v, [5, 25, 50, 75, 95])
        m = float(np.mean(v)); s = float(np.std(v, ddof=1))
        skew = float(np.mean(((v - m) / max(s, 1e-12)) ** 3)) if s > 0 else 0.0
        kurt = float(np.mean(((v - m) / max(s, 1e-12)) ** 4)) - 3.0 if s > 0 else 0.0
        out.append({
            "name": c,
            "label": TRAIT_LABELS.get(c, c),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "mean": float(m),
            "sd": s,
            "unique": int(len(np.unique(v))),
            "skew": skew,
            "kurtosis": kurt,
            "p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4]),
            "provenance": TRAIT_PROVENANCE.get(c, {
                "status": "UNCLASSIFIED",
                "source": "unknown",
                "n": None,
                "note": "No provenance record. Report this.",
            }),
        })
    return out

@app.post("/api/design")
async def design(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "invalid JSON body")
    copula_ops = body.get("copula_ops") or []
    shocks = body.get("shocks") or []
    N = int(body.get("N", 8000))
    T = int(body.get("T", 500))
    seeds = body.get("seeds") or [42]
    seeds = [int(s) for s in seeds][:20]  # cap at 20 seeds
    if not (100 <= N <= 32000): raise HTTPException(400, "N must be 100..32000")
    if not (50 <= T <= 500):    raise HTTPException(400, "T must be 50..500")
    if not seeds:               raise HTTPException(400, "at least one seed required")
    try:
        result = _run_design(copula_ops, shocks, N, T, seeds)
    except Exception as e:
        raise HTTPException(500, f"design failed: {e}")
    benchmarks = _benchmarks_block(result["mean"])
    # Enrich each benchmark with the per-seed std as an "SE over seeds"
    for b, sd in zip(benchmarks, result["std_over_seeds"]):
        b["se_over_seeds"] = float(sd)
    return {
        "scenario": "design",
        "N": N, "T": T, "seeds": seeds, "n_seeds": len(seeds),
        "copula_ops": copula_ops,
        "shocks": shocks,
        "wall_total_s": result["wall_total_s"],
        "platform": result["mean"],
        "per_seed": result["per_seed"],
        "std_over_seeds": result["std_over_seeds"],
        "benchmarks": benchmarks,
        "n_pass": sum(b["pass"] for b in benchmarks),
        "timeseries_full": result["timeseries_full"],
        "timeseries_shock": result["timeseries_shock"],
        "subgroups": result["subgroups"],
        "shock_tick_ranges": [{"tick": int(s["tick"]), "duration": int(s["duration"])}
                              for s in (shocks or [])],
        "run_id": result.get("run_id"),
        "n_agents": result.get("n_agents"),
        "audit": {
            "copula_ops": copula_ops, "shocks": shocks,
            "N": N, "T": T, "seeds": seeds,
            "wall_total_s": result["wall_total_s"],
        },
    }

@app.post("/api/run/{name}")
def run_named(
    name: str,
    N: int = Query(8000, ge=100, le=32000),
    T: int = Query(250, ge=50, le=500),
    seed: int = Query(42),
):
    if name not in SCENARIOS:
        raise HTTPException(404, f"scenario '{name}' not found")
    spec = SCENARIOS[name]
    try:
        result = _run_design([], spec["shocks"], N, T, [seed])
        # If this scenario uses any shock, also run a matched baseline
        # (same seed, same N, same T) so the user can see the delta.
        baseline_result = None
        if spec["shocks"]:
            baseline_result = _run_design([], [], N, T, [seed])
    except Exception as e:
        raise HTTPException(500, f"run failed: {e}")
    benchmarks = _benchmarks_block(result["mean"])
    return {
        "scenario": name,
        "description": spec["description"],
        "N": N, "T": T, "seed": seed,
        "wall_s": result["wall_total_s"],
        "platform": result["mean"],
        "benchmarks": benchmarks,
        "n_pass": sum(b["pass"] for b in benchmarks),
        "timeseries_full": result.get("timeseries_full"),
        "timeseries_shock": result.get("timeseries_shock"),
        "subgroups": result.get("subgroups"),
        "baseline_platform": baseline_result["mean"] if baseline_result else None,
        "shock_tick_ranges": [{"tick": int(s["tick"]), "duration": int(s["duration"])}
                              for s in spec["shocks"]],
        "run_id": result.get("run_id"),
        "n_agents": result.get("n_agents"),
        "audit": {
            "copula": str(os.environ.get("P6C_PARAMS")),
            "shocks": spec["shocks"],
            "N": N, "T": T, "seed": seed,
            "wall_s": result["wall_total_s"],
        },
    }

@app.post("/api/run-custom")
async def run_custom(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "invalid JSON body")
    shocks = body.get("shocks") or []
    if not shocks:
        raise HTTPException(400, "shocks list is empty")
    N = int(body.get("N", 8000))
    T = int(body.get("T", 500))
    seed = int(body.get("seed", 42))
    if not (100 <= N <= 32000): raise HTTPException(400, "N must be 100..32000")
    if not (50 <= T <= 500):    raise HTTPException(400, "T must be 50..500")
    try:
        result = _run_design([], shocks, N, T, [seed])
    except Exception as e:
        raise HTTPException(500, f"run failed: {e}")
    return {
        "scenario": "custom",
        "N": N, "T": T, "seed": seed,
        "wall_s": result["wall_total_s"],
        "platform": result["mean"],
        "benchmarks": _benchmarks_block(result["mean"]),
        "n_pass": sum(b["pass"] for b in _benchmarks_block(result["mean"])),
    }

@app.get("/api/trace/{name}/{agent_i}")
def trace(
    name: str, agent_i: int,
    tick: int = Query(250, ge=0, le=499),
    N: int = Query(8000, ge=100, le=32000),
    T: int = Query(250, ge=50, le=500),
    seed: int = Query(42),
):
    if name not in SCENARIOS:
        raise HTTPException(404, f"scenario '{name}' not found")
    if not (0 <= agent_i < N): raise HTTPException(400, f"agent_i must be in [0, {N})")
    if tick >= T:              raise HTTPException(400, f"tick must be < {T}")
    try:
        tr = trace_in_scenario(name, agent_i=agent_i, N=N, T=T, seed=seed)
    except Exception as e:
        raise HTTPException(500, f"trace failed: {e}")
    row = np.asarray(tr["values"])[tick]
    idx = {k: i for i, k in enumerate(tr["keys"])}
    return {
        "scenario": name, "agent_i": int(agent_i), "tick": int(tick),
        "N": int(N), "T": int(T), "seed": int(seed),
        "shock_state": shock_state_at_tick(tr, tick),
        "fields": {k: float(row[idx[k]]) for k in tr["keys"]},
    }

@app.post("/api/download/{kind}")
async def download(kind: str, request: Request):
    """Echo the POST body back with Content-Disposition. Safari and
    embedded webviews refuse blob: downloads from localhost; a real
    HTTP response always works."""
    body = await request.body()
    if kind == "json":
        return Response(
            content=body, media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="xai32k-result.json"'},
        )
    if kind == "csv":
        return Response(
            content=body, media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="xai32k-result.csv"'},
        )
    raise HTTPException(400, "kind must be 'json' or 'csv'")


@app.get("/api/trace-summary")
def trace_summary(
    scenario: str = Query(...),
    agent_i: int = Query(-1),
    N: int = Query(8000, ge=100, le=32000),
    T: int = Query(250, ge=50, le=500),
    seed: int = Query(42),
):
    """Return per-agent and population-level interaction counts for a
    scenario. Interaction is rare in this engine (~6% baseline, lower
    under stress); this endpoint lets the UI show the base rate and
    offer to jump to a tick where an interaction actually happened."""
    if scenario not in SCENARIOS:
        raise HTTPException(404, f"scenario '{scenario}' not found")
    spec = SCENARIOS[scenario]
    shocks = spec["shocks"]
    import importlib.util, json as _json, time as _time
    p6c_path = WORK / "p6c_v5.py"
    s = importlib.util.spec_from_file_location("p6c_summary", p6c_path)
    p6c = importlib.util.module_from_spec(s)
    sys.modules["p6c_summary"] = p6c
    s.loader.exec_module(p6c)
    import jax.numpy as jnp
    theta = jnp.array([_json.load(open(WORK / "p6c_results.json"))["calib"][k]
                       for k in p6c.CALIB_NAMES], jnp.float32)
    _, SIM = p6c.build_simulator(N, T, shocks=shocks, return_diag=True)
    m, cap, wealth = SIM(theta, seed_val=int(seed))
    cap = np.asarray(cap)
    # col 2 = interacted and is responder; col 3 = interacted and is proposer
    inter = (cap[:, :, 2] > 0) | (cap[:, :, 3] > 0)     # (T, N)
    n_inter = inter.sum(axis=0)                           # (N,)
    order = np.argsort(n_inter)[::-1][:10]
    top_agents = [{"agent_i": int(i), "n_interactions": int(n_inter[i])}
                  for i in order]
    out = {
        "scenario": scenario,
        "N": int(N), "T": int(T), "seed": int(seed),
        "population": {
            "mean":   float(n_inter.mean()),
            "median": float(np.median(n_inter)),
            "p95":    float(np.percentile(n_inter, 95)),
            "max":    int(n_inter.max()),
            "min":    int(n_inter.min()),
            "n_agents_with_zero": int((n_inter == 0).sum()),
            "top_agents": top_agents,
        },
    }
    if 0 <= agent_i < int(N):
        ticks = np.where(inter[:, agent_i])[0].tolist()
        out["agent"] = {
            "agent_i": int(agent_i),
            "n_interactions": int(n_inter[agent_i]),
            "interaction_rate": float(n_inter[agent_i] / max(1, int(T))),
            "interaction_ticks": [int(t) for t in ticks],
        }
    return out



# Run cache: keeps the exact agent matrices used in recent runs, keyed
# by a run_id that is the hash of (copula_ops, N, seed). Lets the 3D
# views show the ACTUAL agents of a run, not a random copula sample.
_RUN_CACHE = {}
_RUN_CACHE_ORDER = []
_RUN_CACHE_MAX = 5


def _run_id_for(copula_ops, N, seed, shocks=None):
    """Hash of everything that determines what agents the engine sees
    AND what happens to them. Two runs with the same copula, N, and seed
    but different shocks MUST produce different ids, because their
    outcomes differ."""
    import hashlib, json as _json
    key = _json.dumps(
        {
            "ops": copula_ops or [],
            "N": int(N),
            "seed": int(seed),
            "shocks": shocks or [],
        },
        sort_keys=True, default=str,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _store_run(run_id, df):
    if run_id in _RUN_CACHE:
        try:
            _RUN_CACHE_ORDER.remove(run_id)
        except ValueError:
            pass
    _RUN_CACHE[run_id] = df
    _RUN_CACHE_ORDER.append(run_id)
    while len(_RUN_CACHE_ORDER) > _RUN_CACHE_MAX:
        old = _RUN_CACHE_ORDER.pop(0)
        _RUN_CACHE.pop(old, None)


def _get_run_df(run_id):
    return _RUN_CACHE.get(run_id)


_POP_CACHE = {}

def _get_baseline_copula():
    """Cache the baseline copula so the crowd tab is instant.
    Uses P6C_PARAMS when set (which is how the .app bundle passes the
    copula path). Falls back to DEFAULT_COPULA only for the dev box."""
    if "df" not in _POP_CACHE:
        import pandas as pd
        path = os.environ.get("P6C_PARAMS", str(DEFAULT_COPULA))
        df = pd.read_pickle(path)
        _POP_CACHE["df"] = df
    return _POP_CACHE["df"]


@app.get("/api/population")
def population(
    trait: str = Query(...),
    n_bins: int = Query(30, ge=10, le=80),
    sort_by: str = Query("id"),
    sort_dir: str = Query("asc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=10, le=200),
    filter_trait: str = Query(""),
    filter_op: str = Query(">="),
    filter_value: float = Query(0.0),
    filter_active: bool = Query(False),
):
    """Return a slice of the baseline copula plus a histogram of the
    selected trait. Sorting and filtering are done on the whole
    population; pagination is applied at the end."""
    df = _get_baseline_copula()
    if trait not in df.columns:
        raise HTTPException(400, f"trait '{trait}' not in copula")

    # histogram
    col = df[trait].to_numpy().astype(float)
    counts, edges = np.histogram(col, bins=int(n_bins))
    histogram = {
        "bins": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
        "mean": float(col.mean()),
        "median": float(np.median(col)),
        "sd": float(col.std(ddof=1)),
        "min": float(col.min()),
        "max": float(col.max()),
        "p25": float(np.percentile(col, 25)),
        "p75": float(np.percentile(col, 75)),
    }

    # filter
    mask = np.ones(len(df), dtype=bool)
    filter_count = int(len(df))
    if filter_active and filter_trait and filter_trait in df.columns:
        fc = df[filter_trait].to_numpy().astype(float)
        op = filter_op
        if op == ">=":   mask = fc >= filter_value
        elif op == ">":  mask = fc >  filter_value
        elif op == "<=": mask = fc <= filter_value
        elif op == "<":  mask = fc <  filter_value
        else:            mask = np.ones(len(df), dtype=bool)
        filter_count = int(mask.sum())

    sub = df.loc[mask].copy()
    sub.insert(0, "id", sub.index)

    # sort
    if sort_by == "id":
        sub = sub.sort_values("id", ascending=(sort_dir == "asc"))
    elif sort_by in sub.columns:
        sub = sub.sort_values(sort_by, ascending=(sort_dir == "asc"))
    else:
        sub = sub.sort_values("id")

    # paginate
    total = int(len(sub))
    start = (page - 1) * page_size
    end = start + page_size
    page_rows = sub.iloc[start:end]

    # return rows as list of dicts, only the columns we care about plus all
    keep_cols = ["id"] + [c for c in sub.columns if c != "id"]
    rows = page_rows[keep_cols].to_dict(orient="records")

    return {
        "trait": trait,
        "histogram": histogram,
        "filter": {
            "active": bool(filter_active and filter_trait),
            "trait": filter_trait, "op": filter_op, "value": float(filter_value),
            "count": filter_count,
        },
        "total": total,
        "page": int(page),
        "page_size": int(page_size),
        "n_pages": max(1, (total + page_size - 1) // page_size),
        "columns": keep_cols,
        "rows": rows,
    }


@app.get("/api/correlations")
def correlations(traits: str = Query("O,C,E,A,N,S_M,S_P,S_N,DASS_Dep,DASS_Anx,DASS_Str,general_distress,m_t,omega_social,theta_T,u_U")):
    """Pearson correlation matrix for the listed copula columns."""
    df = _get_baseline_copula()
    names = [t for t in traits.split(",") if t in df.columns]
    if len(names) < 2:
        raise HTTPException(400, "need at least two valid trait names")
    sub = df[names].to_numpy().astype(float)
    # correlation matrix with NaN-safe handling
    m = np.corrcoef(sub, rowvar=False)
    m = np.nan_to_num(m, nan=0.0)
    return {"traits": names, "matrix": m.tolist()}


@app.get("/api/scenarios-with-meta")
def scenarios_with_meta():
    """Every scenario with description and a human-readable shock summary."""
    out = []
    for name, spec in SCENARIOS.items():
        shocks = spec.get("shocks", [])
        summary = []
        for s in shocks:
            stype = s.get("type", "?")
            mag = s.get("magnitude", 0)
            tick = s.get("tick", 0)
            dur = s.get("duration", 0)
            summary.append(f"{stype} mag={mag} t{tick}-{tick+dur}")
        out.append({
            "name": name,
            "description": spec.get("description", ""),
            "n_shocks": len(shocks),
            "shock_summary": "; ".join(summary),
        })
    return out


@app.get("/api/runs")
def list_runs():
    """List runs currently held in the in-memory cache, with metadata."""
    out = []
    for rid in _RUN_CACHE_ORDER:
        df = _RUN_CACHE.get(rid)
        if df is None:
            continue
        out.append({
            "run_id": rid,
            "n_agents": int(len(df)),
            "has_wealth": "final_wealth" in df.columns,
        })
    return {"runs": out, "max_cache": _RUN_CACHE_MAX}


@app.get("/api/scatter3d")
def scatter3d(
    run_id: str = Query(...),
    x: str = Query("O"),
    y: str = Query("C"),
    z: str = Query("E"),
    color: str = Query("N"),
    max_points: int = Query(2500, ge=200, le=8000),
    highlight: int = Query(-1, ge=-1),
    reference_run_id: str = Query(""),
):
    """Return 3-axis trait coordinates for agents actually used in a run.

    If reference_run_id is provided, the axes are z-scored against the
    reference run's population — so two clouds can be compared on the
    same coordinate system in the 3D comparison view.
    """
    df = _get_run_df(run_id)
    if df is None:
        raise HTTPException(404, "run_id expired or unknown; run a scenario to refresh")
    for name in (x, y, z, color):
        if name not in df.columns:
            raise HTTPException(400, f"trait '{name}' not in run's copula")

    N = len(df)

    # Which population provides the z-score stats?
    ref_df = _get_run_df(reference_run_id) if reference_run_id else None
    stat_df = ref_df if ref_df is not None else df
    for name in (x, y, z, color):
        if name not in stat_df.columns:
            stat_df = df  # fall back to self-stats
            break

    def zscore(col, data=None):
        src = data if data is not None else df
        v = src[col].to_numpy().astype(np.float64)
        mu, sd = v.mean(), max(v.std(ddof=1), 1e-9)
        v2 = df[col].to_numpy().astype(np.float64)
        return (v2 - mu) / sd

    xs_all = zscore(x, stat_df)
    ys_all = zscore(y, stat_df)
    zs_all = zscore(z, stat_df)
    cs_all = zscore(color, stat_df)

    # sample for display; force-include the highlight index
    rng = np.random.default_rng(0)
    if N > max_points:
        idx = rng.choice(N, size=max_points, replace=False).astype(int)
        if 0 <= highlight < N and highlight not in idx:
            idx[0] = int(highlight)
    else:
        idx = np.arange(N, dtype=int)

    cs = cs_all[idx]
    # color scale: if comparing, use the reference's own range so both
    # clouds share a color scale; otherwise use this run's own range
    if ref_df is not None and color in ref_df.columns:
        ref_v = ref_df[color].to_numpy().astype(np.float64)
        ref_mu, ref_sd = ref_v.mean(), max(ref_v.std(ddof=1), 1e-9)
        ref_z = (ref_v - ref_mu) / ref_sd
        cmin, cmax = float(ref_z.min()), float(ref_z.max())
    else:
        cmin, cmax = float(cs.min()), float(cs.max())
    csn = (cs - cmin) / max(cmax - cmin, 1e-9)

    pts = list(zip(
        xs_all[idx].tolist(), ys_all[idx].tolist(), zs_all[idx].tolist(),
        csn.tolist(), idx.tolist(),
    ))

    highlight_point = None
    if 0 <= highlight < N:
        highlight_point = {
            "agent_i": int(highlight),
            "x": float(xs_all[highlight]),
            "y": float(ys_all[highlight]),
            "z": float(zs_all[highlight]),
        }

    return {
        "run_id": run_id,
        "n_agents_total": int(N),
        "n_agents_sampled": int(len(idx)),
        "axes": {"x": x, "y": y, "z": z, "color": color},
        "color_range": [cmin, cmax],
        "points": pts,
        "highlight": highlight_point,
    }


@app.get("/api/agent-profile")
def agent_profile(
    run_id: str = Query(...),
    agent_i: int = Query(..., ge=0),
):
    """Return z-scored traits for a single agent as used in a specific
    run. Z-scored against the run's own population, so a shifted
    population reads correctly. agent_i is the run-local index (0..N-1)."""
    df = _get_run_df(run_id)
    if df is None:
        raise HTTPException(404, "run_id expired or unknown; run a scenario to refresh")
    N = len(df)
    if not (0 <= agent_i < N):
        raise HTTPException(400, f"agent_i must be in [0, {N}) for this run")

    traits = ["O", "C", "E", "A", "N",
              "S_M", "S_P", "S_N", "D",
              "DASS_Dep", "DASS_Anx", "DASS_Str", "general_distress",
              "m_t", "omega_social", "theta_T"]
    labels = {"O": "Open", "C": "Consc", "E": "Extra", "A": "Agree",
              "N": "Neuro", "S_M": "Mach", "S_P": "Psy", "S_N": "Narc",
              "D": "Dark", "DASS_Dep": "Dep", "DASS_Anx": "Anx",
              "DASS_Str": "Str", "general_distress": "Distr",
              "m_t": "Mood", "omega_social": "Soc", "theta_T": "Thr"}

    values = []
    for t in traits:
        if t not in df.columns:
            values.append({"trait": t, "label": labels.get(t, t), "z": 0.0})
            continue
        v = df[t].to_numpy().astype(np.float64)
        z = (v[agent_i] - v.mean()) / max(v.std(ddof=1), 1e-9)
        values.append({"trait": t, "label": labels.get(t, t), "z": float(z)})

    return {
        "run_id": run_id,
        "agent_i": int(agent_i),
        "n_agents": int(N),
        "values": values,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (WORK / "dashboard_ui.html").read_text(encoding="utf-8")
