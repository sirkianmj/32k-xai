import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import hashlib, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from engine_v2 import run_simulation, WS_N

OUT_DIR    = Path(__file__).parent
PARAMS_PKL = Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"
N_TICKS    = 500
SEED       = 42

def state_hash(Q_mf, Q_td, wealth, distress):
    h = hashlib.sha256()
    for a in (Q_mf, Q_td, wealth, distress):
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()

if __name__ == "__main__":
    print(f"[run_32k_v2] params={PARAMS_PKL}")
    metrics, final_state, wall = run_simulation(PARAMS_PKL, N_TICKS, seed=SEED)

    df = pd.DataFrame({
        'tick':             np.arange(N_TICKS),
        'cooperation_rate': metrics['coop_rate'],
        'n_mutual_pairs':   metrics['n_mutual_pairs'],
        'mean_distress':    metrics['mean_distress'],
        'mean_w_mb_eff':    metrics['mean_w_mb_eff'],
        'defection_count':  metrics['defection_count'],
        'wealth_gini':      metrics['gini'],
        'mean_abs_Q_net':   metrics['mean_abs_Q_net'],
        'mean_reward':      metrics['mean_reward'],
    })
    df.to_csv(OUT_DIR / "run_32k_v2.log", index=False)
    print(f"[run_32k_v2] wrote run_32k_v2.log")

    with open(OUT_DIR / "final_state_v2.pkl", "wb") as f:
        pickle.dump({**final_state, 'seed': SEED, 'n_ticks': N_TICKS}, f)
    print(f"[run_32k_v2] wrote final_state_v2.pkl")

    h = state_hash(final_state['Q_mf'], final_state['Q_td'],
                   final_state['wealth'], final_state['distress'])

    with open(OUT_DIR / "run_32k_v2_summary.txt", "w") as f:
        f.write(f"n_agents={WS_N}\nn_ticks={N_TICKS}\nseed={SEED}\n")
        f.write(f"wall_time_s={wall:.4f}\n")
        f.write(f"state_hash={h}\n")
        for k in ['coop_rate', 'gini', 'mean_distress', 'mean_w_mb_eff',
                  'defection_count', 'mean_abs_Q_net', 'mean_reward']:
            f.write(f"final_{k}={metrics[k][-1]:.6f}\n")

    print(f"[run_32k_v2] state_hash      = {h}")
    print(f"[run_32k_v2] wall_time_s     = {wall:.3f}")
    print(f"[run_32k_v2] final coop_rate = {metrics['coop_rate'][-1]:.4f}")
    print(f"[run_32k_v2] final gini      = {metrics['gini'][-1]:.4f}")
    print(f"[run_32k_v2] final distress  = {metrics['mean_distress'][-1]:.4f}")
    print(f"[run_32k_v2] final w_mb_eff  = {metrics['mean_w_mb_eff'][-1]:.4f}")
    print(f"[run_32k_v2] final defection = {metrics['defection_count'][-1]:.0f}")
    print(f"[run_32k_v2] final mean_Qnet = {metrics['mean_abs_Q_net'][-1]:.4f}")
    print(f"[run_32k_v2] final mean_r    = {metrics['mean_reward'][-1]:.4f}")
