"""32k-XAI native window launcher.

Shows a splash screen while uvicorn + JAX compile, then loads the
dashboard. Exposes a Python API for native save dialogs.
"""
import os, sys, time, signal, subprocess, base64, threading
from pathlib import Path
import webview

HERE = Path(__file__).resolve().parent
PORT = 8765
URL = f"http://127.0.0.1:{PORT}"

os.environ.setdefault("JAX_PLATFORMS", "metal")
os.environ.setdefault("P6C_PARAMS",
    str(HERE.parent / "work_g2" / "synthetic_32k_p6d.pkl"))

# ---------- Native save dialog bridge ----------
class Api:
    """Exposed to JavaScript as window.pywebview.api.<method>."""
    def __init__(self):
        self.window = None

    def save_file(self, name, content_b64, file_type="json"):
        """Open the native save dialog and write the given bytes.

        content_b64 is a base64-encoded string of the file contents.
        Returns {"ok": True, "path": "..."} or {"ok": False, "error": "..."}.
        """
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
        # pywebview returns str or tuple depending on platform
        path = target[0] if isinstance(target, (list, tuple)) else target
        try:
            data = base64.b64decode(content_b64)
            Path(path).write_bytes(data)
            return {"ok": True, "path": str(path)}
        except Exception as e:
            return {"ok": False, "error": f"write failed: {e}"}

    def open_report(self, content_b64):
        """Save the report HTML to a temp file and open it in the
        user's default browser, where it can be printed with Cmd+P.
        This is the reliable print path on macOS WKWebView, where
        window.print() from an iframe is not consistently supported.
        """
        import tempfile
        try:
            data = base64.b64decode(content_b64)
            fd, path = tempfile.mkstemp(suffix=".html", prefix="xai32k-report-")
            os.close(fd)
            Path(path).write_bytes(data)
            subprocess.Popen(["open", path])
            return {"ok": True, "path": path}
        except Exception as e:
            return {"ok": False, "error": str(e)}


API = Api()

# ---------- Splash ----------
SPLASH_HTML = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>Loading</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; background: #f6f7f9;
               font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
  .wrap { display: grid; place-items: center; height: 100%; }
  .card { text-align: center; }
  .logo { width: 128px; height: 128px; margin: 0 auto 18px; display: block;
          border-radius: 28px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
  h1 { margin: 0; font-size: 22px; font-weight: 600; letter-spacing: -0.02em;
       color: #1c1c1e; }
  p  { color: #8e8e93; font-size: 13px; margin-top: 6px; }
  .dots { display: inline-block; width: 20px; text-align: left; }
  .dots::after { content: '...'; animation: dots 1.4s steps(4, end) infinite;
                 overflow: hidden; display: inline-block; vertical-align: bottom; }
  @keyframes dots { 0% {width: 0} 100% {width: 1.5em} }
</style></head>
<body>
  <div class="wrap"><div class="card">
    <img class="logo" src="__ICON_DATA_URI__" alt="32k-XAI">
    <h1>XAI-32k Cognitive Simulation</h1>
    <p>Starting up<span class="dots"></span></p>
  </div></div>
</body></html>"""

ERROR_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Startup error</title>
<style>body{font-family:-apple-system,sans-serif;padding:40px;color:#1c1c1e}
pre{background:#f6f7f9;padding:14px;border-radius:8px;overflow:auto;font-size:12px}</style>
</head><body>
<h2>Startup failed</h2>
<p>The engine did not respond in time. Log file:</p>
<pre>__LOGPATH__</pre>
<p>Try quitting and reopening the app. If it persists, check the log file.</p>
</body></html>"""

def _icon_data_uri():
    p = HERE.parent.parent / "static" / "icon-128.png"
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _wait_for_health(timeout=90):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{URL}/api/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def main():
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "dashboard:app",
         "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
        cwd=str(HERE), env=env)

    splash = SPLASH_HTML.replace("__ICON_DATA_URI__", _icon_data_uri())
    window = webview.create_window(
        "XAI-32k Cognitive Simulation",
        html=splash,
        js_api=API,
        width=1400, height=900, min_size=(900, 600),
        text_select=True,
        background_color="#f6f7f9",
    )
    API.window = window

    def background():
        if _wait_for_health():
            window.load_url(URL)
        else:
            log_path = Path.home() / "Library" / "Logs" / "XAI-32k" / "server.log"
            window.load_html(ERROR_HTML.replace("__LOGPATH__", str(log_path)))

    try:
        webview.start(background, debug=False)
    finally:
        proc.send_signal(signal.SIGTERM)
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: proc.kill()


if __name__ == "__main__":
    main()
