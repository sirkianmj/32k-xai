import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import hashlib, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from engine import run_simulation, WS_N

OUT_DIR    = Path(__file__).parent
PARAMS_PKL = Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"
N_TICKS    = 500
SEED       = 42

def state_hash(wealth, distress):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(wealth,   dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(distress, dtype=np.float32).tobytes())
    return h.hexdigest()

if __name__ == "__main__":
    print(f"[run_32k] params={PARAMS_PKL}")
    metrics, final_state, wall = run_simulation(PARAMS_PKL, N_TICKS, seed=SEED)

    log_path = OUT_DIR / "run_32k.log"
    df = pd.DataFrame({
        'tick':             np.arange(N_TICKS),
        'cooperation_rate': metrics['coop_rate'],
        'n_mutual_pairs':   metrics['n_mutual_pairs'],
        'mean_distress':    metrics['mean_distress'],
        'mean_w_mb_eff':    metrics['mean_w_mb_eff'],
        'defection_count':  metrics['defection_count'],
        'wealth_gini':      metrics['gini'],
        'n_accept':         metrics['n_accept'],
        'n_responder':      metrics['n_responder'],
    })
    df.to_csv(log_path, index=False)
    print(f"[run_32k] wrote {log_path}")

    state_path = OUT_DIR / "final_state.pkl"
    with open(state_path, "wb") as f:
        pickle.dump({
            'wealth':      final_state['wealth'],
            'distress':    final_state['distress'],
            'last_accept': final_state['last_accept'],
            'seed':        SEED,
            'n_ticks':     N_TICKS,
        }, f)
    print(f"[run_32k] wrote {state_path}")

    h = state_hash(final_state['wealth'], final_state['distress'])
    summary_path = OUT_DIR / "run_32k_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"n_agents={WS_N}\n")
        f.write(f"n_ticks={N_TICKS}\n")
        f.write(f"seed={SEED}\n")
        f.write(f"wall_time_s={wall:.4f}\n")
        f.write(f"state_hash={h}\n")
        f.write(f"final_coop_rate={metrics['coop_rate'][-1]:.6f}\n")
        f.write(f"final_gini={metrics['gini'][-1]:.6f}\n")
        f.write(f"final_mean_distress={metrics['mean_distress'][-1]:.6f}\n")
        f.write(f"final_mean_w_mb_eff={metrics['mean_w_mb_eff'][-1]:.6f}\n")
        f.write(f"final_defection_count={metrics['defection_count'][-1]:.0f}\n")

    print(f"[run_32k] state_hash       = {h}")
    print(f"[run_32k] wall_time_s      = {wall:.3f}")
    print(f"[run_32k] final coop_rate  = {metrics['coop_rate'][-1]:.4f}")
    print(f"[run_32k] final gini       = {metrics['gini'][-1]:.4f}")
    print(f"[run_32k] final distress   = {metrics['mean_distress'][-1]:.4f}")
    print(f"[run_32k] final w_mb_eff   = {metrics['mean_w_mb_eff'][-1]:.4f}")
    print(f"[run_32k] final defection  = {metrics['defection_count'][-1]:.0f}")
    print(f"[run_32k] wrote {summary_path}")
