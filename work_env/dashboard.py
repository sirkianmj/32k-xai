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
    """Run the same design across a list of seeds; return mean, per-seed, and audit."""
    # 1. Build modified copula
    df = _load_copula()
    if copula_ops:
        df = _apply_copula_ops(df, copula_ops)
    # 2. Subset to N rows (deterministic based on first seed)
    rng = np.random.default_rng(int(seeds[0]))
    idx = rng.integers(0, len(df), size=int(N))
    df = df.iloc[idx].reset_index(drop=True)
    # 3. Resolve target_where masks into target_mask
    resolved_shocks = []
    for s in (shocks or []):
        s2 = dict(s)
        mask = _resolve_target_mask(df, s)
        if mask is not None:
            s2["target_mask"] = mask
            s2.pop("target_where", None)
        resolved_shocks.append(s2)
    # 4. Write modified copula to a unique temp path
    tmp = Path("/tmp") / f"p6c_design_{uuid.uuid4().hex}.pkl"
    df.to_pickle(tmp)
    try:
        # 5. Run each seed
        per_seed = []
        walls = []
        for seed in seeds:
            plat, w = _run_once(tmp, resolved_shocks, N, T, seed)
            per_seed.append(plat)
            walls.append(w)
        per_seed_arr = np.array(per_seed)                # (S, 5)
        mean = per_seed_arr.mean(axis=0)
        std = per_seed_arr.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(5)
        return {
            "mean": mean.tolist(),
            "std_over_seeds": std.tolist(),
            "per_seed": per_seed_arr.tolist(),
            "n_seeds": len(seeds),
            "wall_total_s": float(sum(walls)),
        }
    finally:
        try: tmp.unlink()
        except Exception: pass

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
