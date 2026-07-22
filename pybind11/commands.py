from __future__ import annotations

import os
import sys
import sysconfig

DIR = os.path.abspath(os.path.dirname(__file__))


def _get_config_var(name: str, fmt: str = "{}") -> list[str]:
    var = sysconfig.get_config_var(name)
    return [fmt.format(str(var).strip())] if var else []


def get_include(user: bool = False) -> str:  # noqa: ARG001
    """
    Return the path to the pybind11 include directory. The historical "user"
    argument is unused, and may be removed.
    """
    installed_path = os.path.join(DIR, "include")
    source_path = os.path.join(os.path.dirname(DIR), "include")
    return installed_path if os.path.exists(installed_path) else source_path


def get_cmake_dir() -> str:
    """
    Return the path to the pybind11 CMake module directory.
    """
    cmake_installed_path = os.path.join(DIR, "share", "cmake", "pybind11")
    if os.path.exists(cmake_installed_path):
        return cmake_installed_path

    msg = "pybind11 not installed, installation required to access the CMake files"
    raise ImportError(msg)


def get_pkgconfig_dir() -> str:
    """
    Return the path to the pybind11 pkgconfig directory.
    """
    pkgconfig_installed_path = os.path.join(DIR, "share", "pkgconfig")
    if os.path.exists(pkgconfig_installed_path):
        return pkgconfig_installed_path

    msg = "pybind11 not installed, installation required to access the pkgconfig files"
    raise ImportError(msg)


def get_include_dirs() -> list[str]:
    """
    Return the unique include directories for Python and pybind11.
    """
    dirs = [
        sysconfig.get_path("include"),
        sysconfig.get_path("platinclude"),
        get_include(),
    ]

    # Make unique but preserve order
    unique_dirs = []
    for d in dirs:
        if d and d not in unique_dirs:
            unique_dirs.append(d)
    return unique_dirs


def get_cflags() -> str:
    """
    Return the compile flags for building a simple extension with a
    command-line compiler. Based on python-config.
    """
    flags = [f"-I{d}" for d in get_include_dirs()]
    flags += _get_config_var("CFLAGS")
    flags.append("-std=c++17")
    return " ".join(flags)


def get_ldflags(embed: bool = False) -> str:
    """
    Return the link flags for building a simple extension (or, with
    embed=True, an embedding program) with a command-line compiler.
    Based on python-config.
    """
    flags = _get_config_var("LDFLAGS")

    if embed:
        flags += _get_config_var("LIBDIR", "-L{}")
        if not sysconfig.get_config_var("Py_ENABLE_SHARED"):
            flags += _get_config_var("LIBPL", "-L{}")
        version = sysconfig.get_config_var("VERSION") or ""
        abiflags = getattr(sys, "abiflags", "") or ""
        flags.append(f"-lpython{version}{abiflags}")
        flags += _get_config_var("LIBS")
        flags += _get_config_var("SYSLIBS")
    elif sys.platform.startswith("darwin"):
        flags += ["-undefined", "dynamic_lookup", "-shared"]
    elif sys.platform.startswith("linux"):
        flags += ["-fPIC", "-shared"]

    return " ".join(flags)
