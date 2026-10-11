# scenarios.py — Phase 4: named shock packages for reproducible runs.
#
# Each scenario is a list of shock specs (see notes_phase4_shocks.md).
# The run_scenario() function builds a simulator with the scenario's
# shocks, runs it, and returns the five platform benchmarks plus a
# JSON-serialisable audit record.
#
# Usage:
#   from scenarios import SCENARIOS, run_scenario
#   result = run_scenario("resource_crisis", N=32000, T=500, seed=42)
#   print(result["platform"])
#   print(result["audit"])
import os, json, time
from pathlib import Path
from dataclasses import dataclass, asdict, field
import numpy as np
import jax.numpy as jnp

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

# ---- Scenario definitions ----

SCENARIOS = {
    "baseline": {
        "description": "No shocks. Calibrated reference run.",
        "shocks": [],
    },
    "resource_crisis": {
        "description": ("Resource scarcity 0.6 for all agents, "
                        "ticks 200-350. Economic stress condition."),
        "shocks": [
            {"tick": 200, "duration": 150, "type": "resource_scarcity",
             "magnitude": 0.6, "target_frac": None, "target_seed": 0},
        ],
    },
    "widespread_distress": {
        "description": ("Distress injection 0.8 on 70% of agents, "
                        "ticks 150-350. Psychological stress condition."),
        "shocks": [
            {"tick": 150, "duration": 200, "type": "distress_injection",
             "magnitude": 0.8, "target_frac": 0.7, "target_seed": 3},
        ],
    },
    "compound_stress": {
        "description": ("Resource scarcity 0.4 all agents from tick 100, "
                        "plus distress injection 0.6 on 50% from tick 150. "
                        "Simultaneous economic and psychological stress."),
        "shocks": [
            {"tick": 100, "duration": 200, "type": "resource_scarcity",
             "magnitude": 0.4, "target_frac": None, "target_seed": 0},
            {"tick": 150, "duration": 200, "type": "distress_injection",
             "magnitude": 0.6, "target_frac": 0.5, "target_seed": 7},
        ],
    },
}

# ---- Runner ----

@dataclass
class ScenarioAudit:
    scenario_name: str
    description: str
    N: int
    T: int
    seed: int
    copula: str
    theta: dict
    shocks: list
    wall_s: float
    platform: list
    benchmark_names: list = field(default_factory=lambda: [
        "UG mean offer", "UG rejection", "TG mean send",
        "TG mean return", "Final Gini",
    ])

def run_scenario(name, N=32000, T=500, seed=42,
                 copula=None, theta_path=None, verbose=True):
    import importlib.util, sys
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario '{name}'. Available: {list(SCENARIOS)}")
    spec = SCENARIOS[name]

    here = Path(__file__).parent
    if copula is None:
        copula = Path.home() / "projects/xai_project/work_g2/synthetic_32k_p6d.pkl"
    else:
        copula = Path(copula)
    if theta_path is None:
        theta_path = here / "p6c_results.json"

    # Set the copula path in the environment BEFORE importing p6c_v5.
    # p6c_v5 reads P6C_PARAMS at module-execution time to set PARAMS and
    # load _SRC. Setting it after exec_module is too late.
    os.environ["P6C_PARAMS"] = str(copula)

    # Load p6c_v5 as a fresh module
    p6c_path = here / "p6c_v5.py"
    s = importlib.util.spec_from_file_location("p6c_scenario", p6c_path)
    p6c = importlib.util.module_from_spec(s)
    sys.modules["p6c_scenario"] = p6c
    s.loader.exec_module(p6c)

    # Sanity check: the module should have loaded from `copula`, not the default
    loaded_from = str(p6c.PARAMS)
    if loaded_from != str(copula):
        raise RuntimeError(
            f"scenarios.py: copula override failed. "
            f"p6c_v5.PARAMS={loaded_from}  expected={copula}"
        )

    theta_dict = json.load(open(theta_path))["calib"]
    theta = jnp.array([theta_dict[k] for k in p6c.CALIB_NAMES], jnp.float32)

    if verbose:
        print(f"[scenario] {name}: {spec['description']}", flush=True)
        print(f"[scenario] N={N} T={T} seed={seed} shocks={len(spec['shocks'])}", flush=True)

    t0 = time.time()
    _, SIM = p6c.build_simulator(N, T, shocks=spec["shocks"])
    _ = SIM(theta, seed_val=seed).block_until_ready()
    plat = np.asarray(SIM(theta, seed_val=seed))
    wall = time.time() - t0

    if verbose:
        print(f"[scenario] wall={wall:.2f}s platform={plat}", flush=True)

    audit = ScenarioAudit(
        scenario_name=name,
        description=spec["description"],
        N=N, T=T, seed=seed,
        copula=str(copula),
        theta={k: float(v) for k, v in theta_dict.items()},
        shocks=spec["shocks"],
        wall_s=wall,
        platform=[float(x) for x in plat],
    )
    return {"platform": plat, "audit": asdict(audit)}

# ---- CLI ----

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", nargs="?", default=None,
                        help="scenario name, or 'all' to run every scenario")
    parser.add_argument("--N", type=int, default=32000)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="scenario_results.json")
    args = parser.parse_args()

    here = Path(__file__).parent
    out_path = here / args.out

    if args.scenario is None or args.scenario == "all":
        names = list(SCENARIOS.keys())
    else:
        names = [args.scenario]

    results = {}
    for nm in names:
        r = run_scenario(nm, N=args.N, T=args.T, seed=args.seed)
        results[nm] = {
            "platform": [float(x) for x in r["platform"]],
            "audit": r["audit"],
        }
        print()

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[scenarios] wrote {out_path}")
