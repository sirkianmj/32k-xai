import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
from pathlib import Path
import numpy as np
from xai_trace import trace_agent, TRACE_KEYS

CKPT_DIR = Path(__file__).parent / "ckpt_p3_test"
PARAMS = Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"
N_TICKS = 500
AGENTS = [0, 100, 12345]
SAMPLE_TICKS = [0, 50, 200, 449]
CKPT = next(CKPT_DIR.glob("tick_0050.npz"))

if __name__ == "__main__":
    for ai in AGENTS:
        tr = trace_agent(str(CKPT), target_tick=N_TICKS, agent_i=ai, params_pkl=PARAMS)
        v = tr["values"]
        idx = {k: i for i, k in enumerate(TRACE_KEYS)}
        print(f"\n===== agent {ai} | {v.shape[0]} ticks traced =====")
        print(f"{'t':>4}  {'Q_mf0':>7} {'Q_mf1':>7}  {'Q_MB0':>7} {'Q_MB1':>7}  "
              f"{'w_mb':>6} {'dist':>6}  a1 s2 a2  {'reward':>7}  "
              f"{'int':>3} {'rej':>3}  {'p_id':>7} {'p_a2':>4} {'p_acc':>5}")
        for k in SAMPLE_TICKS:
            if k >= v.shape[0]:
                continue
            r = v[k]
            print(f"{tr['start_tick']+k:>4}  "
                  f"{r[idx['Q_mf_a0']]:>7.3f} {r[idx['Q_mf_a1']]:>7.3f}  "
                  f"{r[idx['Q_MB_a0']]:>7.3f} {r[idx['Q_MB_a1']]:>7.3f}  "
                  f"{r[idx['w_mb_eff']]:>6.3f} {r[idx['distress_pre']]:>6.3f}  "
                  f"{int(r[idx['a1']])}  {int(r[idx['s2']])}  {int(r[idx['a2']])}  "
                  f"{r[idx['reward']]:>7.3f}  "
                  f"{int(r[idx['interacting']]):>3} {int(r[idx['rejected']]):>3}  "
                  f"{int(r[idx['partner_id']]):>7} {int(r[idx['partner_a2']]):>4} "
                  f"{int(r[idx['partner_accept']]):>5}")
