from __future__ import annotations

import platform
import sys
import sysconfig

ANDROID = sys.platform.startswith("android")
LINUX = sys.platform.startswith("linux")
MACOS = sys.platform.startswith("darwin")
WIN = sys.platform.startswith("win32") or sys.platform.startswith("cygwin")
FREEBSD = sys.platform.startswith("freebsd")

CPYTHON = platform.python_implementation() == "CPython"
PYPY = platform.python_implementation() == "PyPy"
GRAALPY = sys.implementation.name == "graalpy"
_graalpy_version = (
    sys.modules["__graalpython__"].get_graalvm_version() if GRAALPY else "0.0.0"
)
GRAALPY_VERSION = tuple(int(t) for t in _graalpy_version.split("-")[0].split(".")[:3])

# Compile-time config (what the binary was built for)
PY_GIL_DISABLED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
# Runtime state (what's actually happening now)
sys_is_gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)

TYPES_ARE_IMMORTAL = (
    PYPY
    or GRAALPY
    or (CPYTHON and PY_GIL_DISABLED and (3, 13) <= sys.version_info < (3, 14))
)
