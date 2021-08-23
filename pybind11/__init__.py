# -*- coding: utf-8 -*-

from ._version import __version__, version_info
from .commands import get_cmake_dir, get_include

__all__ = (
    "version_info",
    "__version__",
    "get_include",
    "get_cmake_dir",
)
