# -*- coding: utf-8 -*-

from ._version import version_info, __version__
from .commands import get_include, get_cmake_dir


__all__ = (
    "version_info",
    "__version__",
    "get_include",
    "get_cmake_dir",
)
