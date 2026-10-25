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
hiddenimports = [
    "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto", "uvicorn.loops.asyncio",
    "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl", "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan", "uvicorn.lifespan.on", "uvicorn.lifespan.off",
    "fastapi", "starlette", "pydantic", "anyio", "h11", "httptools",
    "jax", "jaxlib", "jax._src", "numpy", "scipy", "pandas", "networkx", "tqdm",
    "webview", "dashboard",
]

for pkg in ("jax", "jaxlib", "webview"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"[spec] collect_all({pkg}) failed: {e}")

a = Analysis(
    [str(REPO / "packaging" / "macos" / "launch_mac.py")],
    pathex=[str(WORK_ENV)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter", "matplotlib", "PyQt5", "PyQt6", "PySide2", "PySide6",
        "metaljax", "torch", "tensorflow", "IPython", "pytest",
    ],
    noarchive=False,
    optimize=2,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="XAI-32k",
    debug=False,
    strip=True,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(exe, a.binaries, a.datas, strip=True, upx=False, name="XAI-32k")

app = BUNDLE(
    coll,
    name="XAI-32k.app",
    icon=str(REPO / "assets" / "app-icon.icns"),
    bundle_identifier="com.sirkianmj.32k-xai",
    info_plist={
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
        "NSRequiresAquaSystemAppearance": False,
    },
)
