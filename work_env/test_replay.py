# test_replay.py — Phase 2g verification:
#   1. Run 500 ticks with checkpoints every 50.
#   2. Re-run 500 ticks without checkpointing, capture the final hash.
#   3. Verify every checkpoint's forward replay matches the reference hash.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
import time
from pathlib import Path
import jax, jax.numpy as jnp
from jax import random, lax
import numpy as np
from engine_v5 import load_all, build_ws, make_step, P_COMMON, N_ACTIONS, WEALTH_INIT, WS_N
from engine_v5_ckpt import (
    run_with_checkpoints, replay_from_checkpoint, state_hash,
    _init_carry, _prepare,
)

N = WS_N
N_TICKS = 500
SEED = 42
CKPT_DIR = Path(__file__).parent / "ckpt_p2g"
PARAMS = Path(os.environ.get("P6C_PARAMS", Path.home() / "projects/xai_project/work_g2/synthetic_32k.pkl"))

def run_reference(params_pkl, n_ticks, seed, N):
    P_np = load_all(params_pkl)
    P, neighbors, _ = _prepare(P_np, N)
    step_fn = make_step(neighbors, P)
    keys = random.split(random.PRNGKey(seed), n_ticks)
    @jax.jit
    def run_inner(carry, ks):
        return lax.scan(step_fn, carry, ks)
    carry = _init_carry(P, N)
    t0 = time.time()
    carry, _outs = run_inner(carry, keys)
    jax.block_until_ready(carry)
    wall = time.time() - t0
    from engine_v5_ckpt import _carry_to_dict
    return _carry_to_dict(carry), wall

if __name__ == "__main__":
    print("=== Reference run (no checkpoints) ===")
    ref_state, ref_wall = run_reference(PARAMS, N_TICKS, SEED, N)
    ref_hash = state_hash(ref_state)
    print(f"[ref] wall={ref_wall:.3f}s  hash={ref_hash}")
    print()

    print("=== Run with checkpoints ===")
    saved, ckpt_wall = run_with_checkpoints(
        PARAMS, N_TICKS, CKPT_DIR, ckpt_every=50, seed=SEED, N=N)
    print()

    print("=== Verifying checkpoint replay ===")
    all_ok = True
    # checkpoint at tick 500 should match reference exactly
    last = saved[-1]
    print(f"[verify] ckpt t={last['tick']} hash={last['hash'][:16]}...")
    if last['hash'] == ref_hash:
        print(f"[verify] ckpt@500 == reference. PASS")
    else:
        print(f"[verify] ckpt@500 hash != reference. FAIL")
        all_ok = False
    # replay from each earlier checkpoint forward to 500 and compare
    for ck in saved[:-1]:
        _, replayed_hash = replay_from_checkpoint(ck['path'], PARAMS, N_TICKS, N=N)
        status = "PASS" if replayed_hash == ref_hash else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"[verify] replay from t={ck['tick']:>4} -> t={N_TICKS}: hash={replayed_hash[:16]}... {status}")
    print()
    print("=== OVERALL ===")
    print("PASS" if all_ok else "FAIL")
