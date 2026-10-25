# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

REPO = Path(SPECPATH).resolve().parent.parent
WORK_ENV = REPO / "work_env"
WORK_G2  = REPO / "work_g2"

datas = []
for name in [
    "dashboard.py", "dashboard_ui.html", "guide_content.py",
    "p6c_v5.py", "engine_v5.py", "scenarios.py",
    "xai_trace_v2.py", "xai_trace_scenarios.py", "p6c_results.json",
]:
    src = WORK_ENV / name
    if src.exists():
        datas.append((str(src), "app/work_env"))

for name in [
    "synthetic_32k_p6d.pkl",
    "phase4_dose_response.json",
    "wealth_mod_evidence.json",
]:
    src = WORK_G2 / name
    if src.exists():
        datas.append((str(src), "work_g2"))

binaries = []
try:
    binaries += collect_dynamic_libs("jaxlib")
except Exception:
    pass
try:
    binaries += collect_dynamic_libs("webview")
except Exception:
    pass

hiddenimports = [
    "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto", "uvicorn.loops.asyncio",
    "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl", "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan", "uvicorn.lifespan.on", "uvicorn.lifespan.off",
    "fastapi", "starlette", "pydantic", "anyio", "h11", "httptools",
    "jax", "jaxlib", "jax._src", "numpy", "scipy", "pandas", "networkx", "tqdm",
    "webview", "webview.platforms.edgechromium", "webview.platforms.winforms",
    "clr", "pythonnet", "clr_loader", "dashboard",
]

for pkg in ("jax", "jaxlib", "webview", "pythonnet", "clr_loader"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"[spec] collect_all({pkg}) failed: {e}")

a = Analysis(
    [str(REPO / "packaging" / "windows" / "launch_windows.py")],
    pathex=[str(WORK_ENV)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[str(REPO / "packaging" / "windows" / "rthook_pythonnet.py")],
    excludes=["tkinter", "matplotlib", "PyQt5", "PyQt6", "PySide2", "PySide6"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="XAI-32k",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(REPO / "packaging" / "windows" / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="XAI-32k",
)
