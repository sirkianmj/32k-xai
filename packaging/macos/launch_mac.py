"""Mac launcher for 32k-XAI (CPU-only)."""
import os, sys, time, threading
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
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("P6C_PARAMS", str(WORK_G2 / "synthetic_32k_p6d.pkl"))
os.chdir(str(WORK_ENV))
sys.path.insert(0, str(WORK_ENV))

import webview
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
    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

    window = webview.create_window(
        "32k-XAI",
        html="<html><body style='font-family:-apple-system,sans-serif;display:grid;"
             "place-items:center;height:100vh;margin:0;background:#f6f7f9;color:#1c1c1e'>"
             "<div style='text-align:center'><h2>Starting 32k-XAI…</h2>"
             "<p style='color:#8e8e93'>Compiling JAX for the first time takes ~20 seconds.</p>"
             "</div></body></html>",
        width=1400, height=900, min_size=(900, 600),
        text_select=True,
        background_color="#f6f7f9",
    )

    def background():
        if _wait_for_health():
            window.load_url(URL)
        else:
            window.load_html("<html><body style='padding:40px;font-family:-apple-system'>"
                             "<h2>Startup failed</h2><p>Close and try again.</p></body></html>")

    webview.start(background, debug=False)

if __name__ == "__main__":
    main()
