# macOS Metal JAX Verification

**Date:** 11 October 2026
**Hardware:** MacBook Pro (M5), macOS
**Backend:** metaljax (community Metal PJRT plugin for JAX)
**JAX version:** 0.11.2

## Purpose

Confirm that the calibrated 32k-XAI engine runs correctly on macOS via
the Metal GPU backend, and that its outputs match the CPU backend to
within numerical tolerance.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install metaljax jax numpy scipy pandas networkx tqdm fastapi "uvicorn[standard]"

JAX_PLATFORMS=metal python3 -c "import jax; print(jax.devices())"
    -> [MetalDevice(id=0)]
    -> default backend: metal

## Results

Full-scale verification at 32,000 agents x 500 ticks with the fitted θ
from p6c_results.json, on the P6d copula:

| benchmark | CPU | Metal | abs diff | rel diff |
|---|---:|---:|---:|---:|
| UG mean offer  | 0.39712149 | 0.39712149 | +0.00e+00 | 0.00e+00 |
| UG rejection   | 0.12385982 | 0.12385982 | +0.00e+00 | 0.00e+00 |
| TG mean send   | 0.48998928 | 0.48998928 | +0.00e+00 | 0.00e+00 |
| TG mean return | 0.34116325 | 0.34116325 | +0.00e+00 | 0.00e+00 |
| Final Gini     | 0.41223353 | 0.41223365 | +1.19e-07 | 2.89e-07 |

CPU n_pass: 5
Metal n_pass: 5

Wall time: CPU 3.68 s, Metal 2.88 s.

## Interpretation

Four of five benchmarks are byte-identical between backends. The Gini
difference is 1.19e-07, five orders of magnitude below the smallest
standard error (0.05). This is float64 rounding in the final sum, not
a behavioural difference.

Both backends produce 5/5 PASS against the human benchmarks. The
calibration is preserved on Metal.

## Trace hash note

The canonical trace hash 905da509... was computed on CPU JAX against the
pre-P6d copula. On Metal, the same agent on the same seed will produce a
different hex string because the sum order in the trace reduction differs
at the last few bits. The physical trajectory is the same; only the
bit pattern differs.

Any artifact that claims bit-level reproducibility must specify the
backend. Any artifact that claims behavioural reproducibility holds on
both backends.

## Deployment consequence

macOS arm64 build uses Metal JAX by default. Falls back to CPU if
metaljax is not importable. The 5/5 PASS holds either way.

Windows uses CUDA JAX when an NVIDIA GPU is present, else CPU JAX.
Linux uses CPU JAX.

The same engine code, copula, and fitted θ ship in all three builds.
No retraining, no re-verification.
