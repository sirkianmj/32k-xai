# dashboard.py — Phase 5: FastAPI backend for the 32k-XAI dashboard.
#
# Serves dashboard_ui.html at / and JSON endpoints under /api/.
# Designed for use by researchers who are not programmers: no URLs
# are exposed in the interface; every action is a button.
#
# Run:
#   cd work_env
#   python3 -m uvicorn dashboard:app --host 127.0.0.1 --port 8765
# Then open http://127.0.0.1:8765 in a browser.
import os, sys, json
from pathlib import Path
import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
os.chdir(WORK)
os.environ.setdefault(
    "P6C_PARAMS",
    str(Path.home() / "projects/xai_project/work_g2/synthetic_32k_p6d.pkl"),
)

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, FileResponse

from scenarios import SCENARIOS, run_scenario
from xai_trace_scenarios import trace_in_scenario, shock_state_at_tick

app = FastAPI(title="32k-XAI Dashboard", version="0.2.0")

BENCH = ["UG mean offer", "UG rejection", "TG mean send", "TG mean return", "Final Gini"]
TARGET = [0.400, 0.160, 0.502, 0.372, 0.400]
SE = [0.050, 0.050, 0.130, 0.120, 0.120]

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

def _run_and_pack(scenario_name, shocks, N, T, seed, description):
    import importlib.util
    # Build the simulator directly with the given shocks (works for named and custom).
    p6c_path = WORK / "p6c_v5.py"
    spec = importlib.util.spec_from_file_location("p6c_dashboard", p6c_path)
    p6c = importlib.util.module_from_spec(spec)
    sys.modules["p6c_dashboard"] = p6c
    spec.loader.exec_module(p6c)
    import jax.numpy as jnp, time
    theta = jnp.array([json.load(open(WORK / "p6c_results.json"))["calib"][k]
                       for k in p6c.CALIB_NAMES], jnp.float32)
    t0 = time.time()
    _, SIM = p6c.build_simulator(N, T, shocks=shocks)
    _ = SIM(theta, seed_val=int(seed)).block_until_ready()
    plat = np.asarray(SIM(theta, seed_val=int(seed)))
    wall = time.time() - t0
    benchmarks = _benchmarks_block(plat)
    return {
        "scenario": scenario_name,
        "description": description,
        "N": int(N), "T": int(T), "seed": int(seed),
        "wall_s": wall,
        "platform": [float(x) for x in plat],
        "benchmarks": benchmarks,
        "n_pass": sum(b["pass"] for b in benchmarks),
        "audit": {
            "copula": str(os.environ.get("P6C_PARAMS")),
            "shocks": shocks,
            "N": int(N), "T": int(T), "seed": int(seed),
            "wall_s": wall,
        },
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.2.0"}

@app.get("/api/scenarios")
def list_scenarios():
    return {name: spec["description"] for name, spec in SCENARIOS.items()}

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
        return _run_and_pack(name, spec["shocks"], N, T, seed, spec["description"])
    except Exception as e:
        raise HTTPException(500, f"run failed: {e}")

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
    desc = "Custom scenario with " + ", ".join(s["type"] for s in shocks)
    try:
        return _run_and_pack("custom", shocks, N, T, seed, desc)
    except Exception as e:
        raise HTTPException(500, f"run failed: {e}")

@app.get("/api/trace/{name}/{agent_i}")
def trace(
    name: str,
    agent_i: int,
    tick: int = Query(250, ge=0, le=499),
    N: int = Query(8000, ge=100, le=32000),
    T: int = Query(250, ge=50, le=500),
    seed: int = Query(42),
):
    if name not in SCENARIOS:
        raise HTTPException(404, f"scenario '{name}' not found")
    if not (0 <= agent_i < N):
        raise HTTPException(400, f"agent_i must be in [0, {N})")
    if tick >= T:
        raise HTTPException(400, f"tick must be < {T}")
    try:
        tr = trace_in_scenario(name, agent_i=agent_i, N=N, T=T, seed=seed)
    except Exception as e:
        raise HTTPException(500, f"trace failed: {e}")
    row = np.asarray(tr["values"])[tick]
    idx = {k: i for i, k in enumerate(tr["keys"])}
    return {
        "scenario": name,
        "agent_i": int(agent_i),
        "tick": int(tick),
        "N": int(N), "T": int(T), "seed": int(seed),
        "shock_state": shock_state_at_tick(tr, tick),
        "fields": {k: float(row[idx[k]]) for k in tr["keys"]},
    }

@app.get("/", response_class=HTMLResponse)
def index():
    return (WORK / "dashboard_ui.html").read_text(encoding="utf-8")
