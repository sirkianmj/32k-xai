import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
import time
from pathlib import Path
import numpy as np
import jax
import engine_v5 as E
from engine_v5_ckpt import run_with_checkpoints
from xai_trace import trace_agent, TRACE_DIM, TRACE_KEYS

CKPT_DIR = Path(__file__).parent / "ckpt_p3_test"
PARAMS = Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"
N_TICKS = 500
SEED = 42
N = E.WS_N

if __name__ == "__main__":
    print(f"=== 1. Reference run (with checkpoints) N={N} ===")
    saved, _wall = run_with_checkpoints(
        PARAMS, N_TICKS, CKPT_DIR, ckpt_every=50, seed=SEED, N=N)
    ref_hash = saved[-1]["hash"]
    ckpt_50 = next(s for s in saved if s["tick"] == 50)
    print(f"[ref] hash = {ref_hash}")
    print()

    print(f"=== 2. Trace agent 100 from tick 50 to 500 ===")
    t0 = time.time()
    tr = trace_agent(ckpt_50["path"], target_tick=N_TICKS,
                     agent_i=100, params_pkl=PARAMS, N=N)
    wall = time.time() - t0
    print(f"[trace] wall = {wall:.3f}s")
    print(f"[trace] shape: {tr['values'].shape}  (expected ({N_TICKS-50}, {TRACE_DIM}))")
    print(f"[trace] keys[:6] = {TRACE_KEYS[:6]}")
    print(f"[trace] final hash = {tr['final_state_hash']}")
    print(f"[trace] reference  = {ref_hash}")
    ok = (tr['final_state_hash'] == ref_hash)
    print(f"[trace] MATCH: {'PASS' if ok else 'FAIL'}")
    print()

    print(f"=== 3. Trace agent 0 from tick 50 to 500 ===")
    tr2 = trace_agent(ckpt_50["path"], target_tick=N_TICKS,
                      agent_i=0, params_pkl=PARAMS, N=N)
    ok2 = (tr2['final_state_hash'] == ref_hash)
    print(f"[trace2] final hash = {tr2['final_state_hash']}")
    print(f"[trace2] MATCH: {'PASS' if ok2 else 'FAIL'}")
    print()

    print("=== 4. Sanity check on trace values for agent 100 ===")
    v = tr["values"]
    idx = {k: i for i, k in enumerate(TRACE_KEYS)}
    a1 = v[:, idx["a1"]].astype(int)
    s2 = v[:, idx["s2"]].astype(int)
    a2 = v[:, idx["a2"]].astype(int)
    inter = v[:, idx["interacting"]]
    print(f"a1 unique: {sorted(set(a1))} (expect [0, 1])")
    print(f"s2 unique: {sorted(set(s2))} (expect [0, 1])")
    print(f"a2 unique: {sorted(set(a2))} (expect [0,1,2,3])")
    print(f"interacting rate: {inter.mean():.3f}")
    print()
    print("=== OVERALL ===")
    print("PASS" if (ok and ok2) else "FAIL")
