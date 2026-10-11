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
from fastapi.responses import HTMLResponse

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

        return {
            "mean": mean.tolist(),
            "std_over_seeds": std.tolist(),
            "per_seed": per_seed_arr.tolist(),
            "n_seeds": len(seeds),
            "wall_total_s": float(sum(walls)),
            "timeseries_full": ts_full,
            "timeseries_shock": ts_shock,
            "subgroups": subgroups,
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
    df = _load_copula()
    out = []
    for c in df.columns:
        v = df[c].to_numpy().astype(np.float64)
        out.append({
            "name": c,
            "label": TRAIT_LABELS.get(c, c),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)),
            "unique": int(len(np.unique(v))),
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

@app.get("/", response_class=HTMLResponse)
def index():
    return (WORK / "dashboard_ui.html").read_text(encoding="utf-8")
