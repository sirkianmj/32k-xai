"""Linux launcher for 32k-XAI.

Starts uvicorn in a daemon thread, waits for /api/health, then opens
the default browser. Uses webbrowser (stdlib) rather than pywebview
because pywebview on Linux depends on libwebkit2gtk-4.0, which is not
installed by default on most distributions.
"""
import os, sys, time, threading, webbrowser
from pathlib import Path

if getattr(sys, "frozen", False):
    DATA_ROOT = Path(sys._MEIPASS)
    APP_DIR   = DATA_ROOT / "app"
else:
    HERE = Path(__file__).resolve().parent
    REPO = HERE.parent.parent
    APP_DIR = REPO / "work_env"
    DATA_ROOT = REPO

WORK_ENV = APP_DIR / "work_env" if (APP_DIR / "work_env").exists() else APP_DIR
WORK_G2  = (DATA_ROOT / "work_g2") if (DATA_ROOT / "work_g2").exists() else (REPO / "work_g2")

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("P6C_PARAMS", str(WORK_G2 / "synthetic_32k_p6d.pkl"))
os.chdir(str(WORK_ENV))
sys.path.insert(0, str(WORK_ENV))

import uvicorn
from dashboard import app

PORT = 8765
URL  = f"http://127.0.0.1:{PORT}"

def _run_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")

def _wait_for_health(timeout=180):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{URL}/api/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def main():
    print("32k-XAI: starting engine...")
    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

    if _wait_for_health():
        print(f"32k-XAI: ready at {URL}")
        print("32k-XAI: opening in your default browser...")
        webbrowser.open(URL)
        print("32k-XAI: press Ctrl+C to stop.")
    else:
        print("32k-XAI: engine did not respond within 180 seconds.", file=sys.stderr)
        sys.exit(1)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n32k-XAI: shutting down.")

if __name__ == "__main__":
    main()
