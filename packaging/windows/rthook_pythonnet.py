"""PyInstaller runtime hook: configure pythonnet for frozen bundles.

Runs before any `import clr` / `import webview`. Sets the .NET runtime
to netfx (the framework pywebview uses on Windows) and adds the bundled
Python.Runtime.dll directory to the DLL search path so clr_loader can
resolve Python.Runtime.Loader.Initialize.
"""
import os, sys
from pathlib import Path

if getattr(sys, "frozen", False):
    base = Path(sys._MEIPASS)

    # Where PyInstaller put pythonnet's runtime files
    runtime_dir = base / "pythonnet" / "runtime"
    if runtime_dir.exists():
        os.add_dll_directory(str(runtime_dir))
        os.environ["PATH"] = str(runtime_dir) + os.pathsep + os.environ.get("PATH", "")

    # Also add the bundle root, where clr.pyd may live
    os.add_dll_directory(str(base))

    # Force pythonnet to use the .NET Framework runtime
    os.environ.setdefault("PYTHONNET_RUNTIME", "netfx")
