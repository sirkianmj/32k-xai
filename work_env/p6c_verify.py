# Verify the calibrated θ at full scale (32k x 500).
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
import json, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import importlib.util

# Load the build_simulator + loss from p6c_v5
spec = importlib.util.spec_from_file_location("p6c", "p6c_v5.py")
p6c = importlib.util.module_from_spec(spec)
# Prevent __main__ from running
import sys
sys.modules["p6c"] = p6c
spec.loader.exec_module(p6c)

# Rebuild the full-scale simulator using the same module
print("Building 32k x 500 simulator ...", flush=True)
t0 = time.time()
_, SIM_FULL = p6c.build_simulator(32000, 500)
theta = jnp.array([1.5763, 1.6824, 1.6915, 2.2244, 1.7622,
                   0.2799, -0.0445, -0.0285, 3.2817], jnp.float32)
_ = SIM_FULL(theta).block_until_ready()
print(f"  compile wall = {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
plat = np.asarray(SIM_FULL(theta))
print(f"  run wall = {time.time()-t0:.2f}s", flush=True)

TARGETS = np.array([0.40, 0.16, 0.502, 0.372, 0.40])
SE      = np.array([0.05, 0.05, 0.130, 0.120, 0.12])
names   = ["UG mean offer", "UG rejection", "TG mean send", "TG mean return", "Final Gini"]

print("\n" + "="*72)
print("FULL-SCALE VERIFICATION (32k x 500)")
print("="*72)
print(f"{'benchmark':22s} {'platform':>10} {'human':>10} {'SE':>8} {'|diff|':>8} {'verdict':>8}")
print("-"*72)
n_pass = 0
for n, p, t, s in zip(names, plat, TARGETS, SE):
    d = abs(p-t); v = "PASS" if d < s else "FAIL"
    n_pass += int(v=="PASS")
    print(f"{n:22s} {p:>10.4f} {t:>10.4f} {s:>8.4f} {d:>8.4f} {v:>8s}")
print("-"*72)
print(f"OVERALL: {n_pass}/5 PASS")

with open("p6c_final.json", "w") as f:
    json.dump({"theta": [float(v) for v in theta],
               "platform": [float(p) for p in plat],
               "targets": TARGETS.tolist(),
               "n_pass": n_pass}, f, indent=2)
print("Saved: p6c_final.json")
