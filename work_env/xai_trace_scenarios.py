# xai_trace_scenarios.py — Phase 4: trace a specific agent inside a scenario.
#
# Wraps xai_trace_v2.trace_agent() with scenario-specific shock specs from
# scenarios.SCENARIOS. Given a scenario name and an agent index, returns
# the same 56-field trace dict that trace_agent returns, plus the scenario
# metadata.
#
# Usage:
#   from xai_trace_scenarios import trace_in_scenario, format_scenario_tick
#   tr = trace_in_scenario("resource_crisis", agent_i=100)
#   print(format_scenario_tick(tr, tick_idx=250))
import os, sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

# Ensure the trace module loads the P6d copula by default.
os.environ.setdefault("P6C_PARAMS",
    str(Path.home() / "projects/xai_project/work_g2/synthetic_32k_p6d.pkl"))

from xai_trace_v2 import trace_agent, TRACE_V2_KEYS, format_tick
from scenarios import SCENARIOS

def trace_in_scenario(scenario_name, agent_i, N=32000, T=500, seed=42):
    if scenario_name not in SCENARIOS:
        raise KeyError(
            f"unknown scenario '{scenario_name}'. "
            f"Available: {sorted(SCENARIOS.keys())}")
    spec = SCENARIOS[scenario_name]
    tr = trace_agent(agent_i=agent_i, N=N, T=T, seed=seed,
                     shocks=spec["shocks"])
    tr["scenario_name"] = scenario_name
    tr["scenario_description"] = spec["description"]
    return tr

def shock_state_at_tick(tr, tick_idx):
    """Return a dict summarising which shocks are active at tick_idx."""
    out = {"active": [], "resource_mod": 1.0,
           "baseline_mod": 0.0, "keff_reduce_frac": 0.0, "redist": 0.0}
    for s in (tr.get("shocks") or []):
        t0 = int(s["tick"]); t1 = t0 + int(s["duration"])
        if t0 <= tick_idx < t1:
            out["active"].append(s["type"])
            stype = s["type"]; mag = float(s["magnitude"])
            if stype == "resource_scarcity":
                out["resource_mod"] *= (1.0 - mag)
            elif stype == "distress_injection":
                out["baseline_mod"] += mag
            elif stype == "network_thinning":
                out["keff_reduce_frac"] = mag
            elif stype == "wealth_redistribution":
                out["redist"] = mag
    return out

def format_scenario_tick(tr, tick_idx):
    """Print the scenario context followed by the standard trace dump."""
    st = shock_state_at_tick(tr, tick_idx)
    lines = [
        f"=== scenario: {tr['scenario_name']}  |  agent {tr['agent_i']}  |  tick {tick_idx} ===",
        f"  shocks active: {st['active'] or 'none'}",
        f"  resource_mod={st['resource_mod']:.4f}  baseline_mod={st['baseline_mod']:+.4f}  "
        f"keff_reduce={st['keff_reduce_frac']:.2f}  redist={st['redist']:.4f}",
    ]
    lines.append(format_tick(tr, tick_idx))
    return "\n".join(lines)
