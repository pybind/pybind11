from __future__ import annotations

import sys

if sys.version_info < (3, 9):  # noqa: UP036
    msg = "pybind11 does not support Python < 3.9. v3.0 was the last release supporting Python 3.8."
    raise ImportError(msg)


from ._version import __version__, version_info
from .commands import get_cmake_dir, get_include, get_pkgconfig_dir

__all__ = (
    "__version__",
    "get_cmake_dir",
    "get_include",
    "get_pkgconfig_dir",
    "version_info",
)
