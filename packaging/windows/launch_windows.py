"""Windows launcher for 32k-XAI.

Starts uvicorn as a subprocess, waits for /api/health, then opens a
pywebview (WebView2) window. Mirrors launch.py on macOS but uses the
frozen-bundle data directory (sys._MEIPASS) when running as a PyInstaller
.exe.
"""
import os, sys, time, signal, subprocess
from pathlib import Path

# --- locate data root ---------------------------------------------------
if getattr(sys, "frozen", False):
    # PyInstaller onefile: data lives in sys._MEIPASS
    DATA_ROOT = Path(sys._MEIPASS)
    APP_DIR   = DATA_ROOT / "app"
else:
    # Running from source: packaging/windows/ -> repo root
    HERE = Path(__file__).resolve().parent
    REPO = HERE.parent.parent
    APP_DIR = REPO / "work_env"
    DATA_ROOT = REPO

WORK_ENV = APP_DIR / "work_env" if (APP_DIR / "work_env").exists() else APP_DIR
WORK_G2  = (DATA_ROOT / "work_g2") if (DATA_ROOT / "work_g2").exists() else (REPO / "work_g2")

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("P6C_PARAMS", str(WORK_G2 / "synthetic_32k_p6d.pkl"))
os.chdir(str(WORK_ENV))
sys.path.insert(0, str(WORK_ENV))

import webview

PORT = 8765
URL  = f"http://127.0.0.1:{PORT}"


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
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    # uvicorn runs from WORK_ENV so `dashboard:app` resolves
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "dashboard:app",
         "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
        cwd=str(WORK_ENV),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    window = webview.create_window(
        "32k-XAI Cognitive Simulation",
        html="<html><body style='font-family:-apple-system,sans-serif;display:grid;"
             "place-items:center;height:100vh;margin:0;background:#f6f7f9;color:#1c1c1e'>"
             "<div style='text-align:center'><h2 style='font-weight:600'>Starting 32k-XAI…</h2>"
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
            window.load_html(
                "<html><body style='font-family:-apple-system,sans-serif;padding:40px'>"
                "<h2>Startup failed</h2><p>The engine did not respond in time.</p>"
                "<p>Close this window and try again. If it persists, run the .exe "
                "from a terminal to see the error output.</p></body></html>")

    try:
        webview.start(background, debug=False)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
