# 32k-XAI — CPU-only portable image.
# Runs on any system with Docker: Windows, macOS, Linux.
# No NVIDIA GPU required. No CUDA. No PyTorch.

FROM python:3.12-slim

# curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Force JAX to the CPU backend.
# Also match the dev-box memory settings so nothing surprises us.
ENV JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copula path. Overrides the dev-box default in p6c_v5 / dashboard.py
# / xai_trace_* via their os.environ.setdefault calls.
ENV P6C_PARAMS=/app/work_g2/synthetic_32k_p6d.pkl

WORKDIR /app

# Python dependencies. CPU-only JAX.
RUN pip install --no-cache-dir \
      "jax[cpu]" \
      numpy scipy pandas networkx tqdm \
      fastapi "uvicorn[standard]"

# Engine code. Whitelist — do not copy the whole work_env, which
# contains ~60 MB of backups, logs, and superseded scripts.
COPY work_env/engine_v5.py             /app/work_env/engine_v5.py
COPY work_env/p6c_v5.py                /app/work_env/p6c_v5.py
COPY work_env/p6c_results.json         /app/work_env/p6c_results.json
COPY work_env/xai_trace_v2.py          /app/work_env/xai_trace_v2.py
COPY work_env/xai_trace_scenarios.py   /app/work_env/xai_trace_scenarios.py
COPY work_env/scenarios.py             /app/work_env/scenarios.py
COPY work_env/dashboard.py             /app/work_env/dashboard.py
COPY work_env/dashboard_ui.html        /app/work_env/dashboard_ui.html

# Runtime data.
COPY work_g2/synthetic_32k_p6d.pkl     /app/work_g2/synthetic_32k_p6d.pkl
COPY work_g2/phase4_dose_response.json /app/work_g2/phase4_dose_response.json
COPY work_g2/wealth_mod_evidence.json  /app/work_g2/wealth_mod_evidence.json
COPY work_g2/P6D_COMPLETION.md         /app/work_g2/P6D_COMPLETION.md

WORKDIR /app/work_env

EXPOSE 8765

HEALTHCHECK --interval=5s --timeout=3s --start-period=20s --retries=5 \
  CMD curl -fsS http://localhost:8765/api/health || exit 1

CMD ["python3", "-m", "uvicorn", "dashboard:app", \
     "--host", "0.0.0.0", "--port", "8765"]
