# -*- coding: utf-8 -*-
import platform
import sys

LINUX = sys.platform.startswith("linux")
MACOS = sys.platform.startswith("darwin")
WIN = sys.platform.startswith("win32") or sys.platform.startswith("cygwin")

CPYTHON = platform.python_implementation() == "CPython"
PYPY = platform.python_implementation() == "PyPy"

PY2 = sys.version_info.major == 2
