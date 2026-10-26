"""Windows launcher for 32k-XAI.

Runs the Uvicorn server in a background thread, waits for /api/health,
then opens a pywebview (WebView2) window. Uses threading rather than
subprocess to avoid the PyInstaller sys.executable fork-bomb issue.
"""
import base64
import os, sys, time
from pathlib import Path
# Redirect stdout/stderr to a log file on the Desktop for debugging
log_path = Path.home() / "Desktop" / "xai32k_debug.log"
try:
    sys.stdout = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stderr = sys.stdout
    print("=== 32k-XAI Launcher Started ===")
except Exception:
    pass
from pathlib import Path
import threading

# --- locate data root ---------------------------------------------------
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
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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


class Api:
    """Exposed to JavaScript as window.pywebview.api.<method>."""
    def __init__(self):
        self.window = None

    def save_file(self, name, content_b64, file_type="json"):
        log_path = Path.home() / "Desktop" / "xai32k_save_debug.log"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- Save requested: {name} ---\n")
            if self.window is None:
                return {"ok": False, "error": "no window"}
            file_types = {
                "json": ("JSON files (*.json)",),
                "csv":  ("CSV files (*.csv)",),
                "svg":  ("SVG images (*.svg)",),
                "png":  ("PNG images (*.png)",),
                "txt":  ("Text files (*.txt)",),
            }.get(file_type, ("All files (*.*)",))
            try:
                target = self.window.create_file_dialog(
                    webview.SAVE_DIALOG,
                    save_filename=name,
                    file_types=file_types,
                )
            except Exception as e:
                return {"ok": False, "error": f"dialog failed: {e}"}
            if not target:
                return {"ok": False, "error": "cancelled"}
            
            path = target[0] if isinstance(target, (list, tuple)) else target
            path = str(path).strip("\0")  # Fix Windows null-byte bug
            
            try:
                data = base64.b64decode(content_b64)
                Path(path).write_bytes(data)
                return {"ok": True, "path": str(path)}
            except Exception as e:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"WRITE FAILED: {e}\nPath: {path}\n")
                return {"ok": False, "error": f"write failed: {e}"}
        except Exception as outer_e:
            return {"ok": False, "error": f"outer failed: {outer_e}"}

API = Api()

def main():
    # Start Uvicorn in a daemon thread. It will die when the main thread exits.
    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

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
        js_api=API,
    )
    API.window = window

    def background():
        if _wait_for_health():
            window.load_url(URL)
        else:
            window.load_html(
                "<html><body style='font-family:-apple-system,sans-serif;padding:40px'>"
                "<h2>Startup failed</h2><p>The engine did not respond in time.</p>"
                "<p>Close this window and try again.</p></body></html>")

    try:
        webview.start(background, debug=False)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
