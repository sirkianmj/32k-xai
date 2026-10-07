import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
import hashlib, pickle
from pathlib import Path
import numpy as np, pandas as pd
from engine_v5 import run, WS_N

OUT = Path(__file__).parent
PARAMS = Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"
N_TICKS = 500
SEED = 42

def state_hash(d):
    h = hashlib.sha256()
    for k in ["Q_mf","Q_s2","P_hat","wealth","distress"]:
        h.update(np.ascontiguousarray(d[k], np.float32).tobytes())
    return h.hexdigest()

if __name__ == "__main__":
    metrics, final_state = run(PARAMS, N_TICKS, seed=SEED)
    pd.DataFrame({
        'tick': np.arange(N_TICKS),
        'cooperation_rate': metrics['coop_rate'],
        'n_mutual_pairs':  metrics['n_pairs'],
        'mean_distress':   metrics['mean_distress'],
        'mean_w_mb_eff':   metrics['mean_w_mb_eff'],
        'defection_count': metrics['defection'],
        'wealth_gini':     metrics['gini'],
        'mean_abs_Q_net':  metrics['mean_abs_Q_net'],
        'mean_reward':     metrics['mean_reward'],
        'mean_a2':         metrics['mean_a2'],
    }).to_csv(OUT / "run_32k_v5.log", index=False)
    with open(OUT / "final_state_v5.pkl", "wb") as f:
        pickle.dump({**final_state, 'seed': SEED, 'n_ticks': N_TICKS}, f)
    h = state_hash(final_state)
    with open(OUT / "run_32k_v5_summary.txt", "w") as f:
        f.write(f"n_agents={WS_N}\nn_ticks={N_TICKS}\nseed={SEED}\n")
        f.write(f"wall_time_s={metrics['wall_s']:.4f}\nstate_hash={h}\n")
        for k in ['coop_rate','gini','mean_distress','mean_w_mb_eff',
                  'defection','mean_abs_Q_net','mean_reward','mean_a2']:
            f.write(f"final_{k}={metrics[k][-1]:.6f}\n")
    print(f"[run_v5] state_hash      = {h}")
    print(f"[run_v5] wall_s          = {metrics['wall_s']:.3f}")
    print(f"[run_v5] final coop      = {metrics['coop_rate'][-1]:.4f}")
    print(f"[run_v5] final gini      = {metrics['gini'][-1]:.4f}")
    print(f"[run_v5] final distress  = {metrics['mean_distress'][-1]:.4f}")
    print(f"[run_v5] final w_mb_eff  = {metrics['mean_w_mb_eff'][-1]:.4f}")
    print(f"[run_v5] final defection = {metrics['defection'][-1]:.0f}")
    print(f"[run_v5] final Q_net     = {metrics['mean_abs_Q_net'][-1]:.4f}")
    print(f"[run_v5] final reward    = {metrics['mean_reward'][-1]:.4f}")
    print(f"[run_v5] final mean_a2   = {metrics['mean_a2'][-1]:.4f}")
