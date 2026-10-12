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

SCENARIOS = {'baseline': {'description': 'No shocks. Calibrated reference run.', 'shocks': []},
 'resource_crisis': {'description': 'Resource scarcity 0.85 for all agents, ticks 50-450. At the '
                                    'default 250-tick run, the shock covers ticks 50-250, giving '
                                    'the population 200 ticks of scarcity. Rejection rises well '
                                    'past the human SE band.',
                     'shocks': [{'tick': 50,
                                 'duration': 400,
                                 'type': 'resource_scarcity',
                                 'magnitude': 0.85,
                                 'target_frac': None,
                                 'target_seed': 0}]},
 'widespread_distress': {'description': 'Distress injection 1.5 on 70% of agents, ticks 50-450. '
                                        'Modest effect by design - distress affects engagement, '
                                        'not the fairness accept rule, so it moves the behavioural '
                                        'aggregates only weakly.',
                         'shocks': [{'tick': 50,
                                     'duration': 400,
                                     'type': 'distress_injection',
                                     'magnitude': 1.5,
                                     'target_frac': 0.7,
                                     'target_seed': 3}]},
 'compound_stress': {'description': 'Simultaneous resource scarcity 0.6 (all agents) and distress '
                                    'injection 1.0 (50% of agents), both starting at tick 50. '
                                    'Compound economic and psychological stress.',
                     'shocks': [{'tick': 50,
                                 'duration': 400,
                                 'type': 'resource_scarcity',
                                 'magnitude': 0.6,
                                 'target_frac': None,
                                 'target_seed': 0},
                                {'tick': 50,
                                 'duration': 400,
                                 'type': 'distress_injection',
                                 'magnitude': 1.0,
                                 'target_frac': 0.5,
                                 'target_seed': 7}]},
 'social_isolation': {'description': 'Network thinning 0.75: neighbour pool cut by three quarters, '
                                     'ticks 50-450. Rejection and inequality rise sharply.',
                      'shocks': [{'tick': 50,
                                  'duration': 400,
                                  'type': 'network_thinning',
                                  'magnitude': 0.75,
                                  'target_frac': None,
                                  'target_seed': 0}]},
 'mild_isolation': {'description': 'Network thinning 0.35: moderate neighbour reduction.',
                    'shocks': [{'tick': 50,
                                'duration': 400,
                                'type': 'network_thinning',
                                'magnitude': 0.35,
                                'target_frac': None,
                                'target_seed': 0}]},
 'wealth_equalization': {'description': 'Wealth redistribution 0.005 per tick for all agents, '
                                        'ticks 50-450. Visible Gini reduction.',
                         'shocks': [{'tick': 50,
                                     'duration': 400,
                                     'type': 'wealth_redistribution',
                                     'magnitude': 0.005,
                                     'target_frac': None,
                                     'target_seed': 0}]},
 'strong_redistribution': {'description': 'Strong wealth redistribution 0.02 per tick, ticks '
                                          '50-450. Near-total equalization of wealth.',
                           'shocks': [{'tick': 50,
                                       'duration': 400,
                                       'type': 'wealth_redistribution',
                                       'magnitude': 0.02,
                                       'target_frac': None,
                                       'target_seed': 0}]},
 'severe_scarcity': {'description': 'Resource scarcity 0.90 for all agents, ticks 50-450. The '
                                    'population earns one tenth of normal. Causes a four-to-five '
                                    'sigma rise in rejection — the benchmark goes from PASS to '
                                    'FAIL.',
                     'shocks': [{'tick': 50,
                                 'duration': 400,
                                 'type': 'resource_scarcity',
                                 'magnitude': 0.9,
                                 'target_frac': None,
                                 'target_seed': 0}]},
 'total_isolation': {'description': 'Social isolation 0.83 for all agents, ticks 50-450. Neighbour '
                                    'pool cut to roughly one sixth. Rejection and inequality rise '
                                    'sharply, TG return rises.',
                     'shocks': [{'tick': 50,
                                 'duration': 400,
                                 'type': 'network_thinning',
                                 'magnitude': 0.83,
                                 'target_frac': None,
                                 'target_seed': 0}]},
 'radical_redistribution': {'description': 'Wealth redistribution 0.05 per tick for all agents, '
                                           'ticks 50-450. Each tick pulls 5% of the gap to the '
                                           'mean. Collapses Gini to near zero.',
                            'shocks': [{'tick': 50,
                                        'duration': 400,
                                        'type': 'wealth_redistribution',
                                        'magnitude': 0.05,
                                        'target_frac': None,
                                        'target_seed': 0}]},
 'compound_crisis': {'description': 'Simultaneous severe scarcity (0.8) and social isolation '
                                    '(0.6), ticks 100-450. Both economic and structural stress at '
                                    'once.',
                     'shocks': [{'tick': 100,
                                 'duration': 350,
                                 'type': 'resource_scarcity',
                                 'magnitude': 0.8,
                                 'target_frac': None,
                                 'target_seed': 0},
                                {'tick': 100,
                                 'duration': 350,
                                 'type': 'network_thinning',
                                 'magnitude': 0.6,
                                 'target_frac': None,
                                 'target_seed': 0}]},
 'mild_distress_check': {'description': 'Distress injection 2.0 for all agents, ticks 50-450. '
                                        'Deliberately extreme. The benchmark movement is small '
                                        'because distress affects engagement, not offer or accept '
                                        'decisions. Documented as an engine property.',
                         'shocks': [{'tick': 50,
                                     'duration': 400,
                                     'type': 'distress_injection',
                                     'magnitude': 2.0,
                                     'target_frac': None,
                                     'target_seed': 0}]}}

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
