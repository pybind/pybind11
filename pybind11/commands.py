from __future__ import annotations

import os
import re
import sys
import sysconfig

DIR = os.path.abspath(os.path.dirname(__file__))

# This is the conditional used for os.path being posixpath
if "posix" in sys.builtin_module_names:
    import shlex

    def _quote(s: str) -> str:
        return shlex.quote(s)
elif "nt" in sys.builtin_module_names:
    # See https://github.com/mesonbuild/meson/blob/db22551ed9d2dd7889abea01cc1c7bba02bf1c75/mesonbuild/utils/universal.py#L1092-L1121
    # and the original documents:
    # https://docs.microsoft.com/en-us/cpp/c-language/parsing-c-command-line-arguments and
    # https://blogs.msdn.microsoft.com/twistylittlepassagesallalike/2011/04/23/everyone-quotes-command-line-arguments-the-wrong-way/
    _UNSAFE = re.compile("[ \t\n\r]")

    def _quote(s: str) -> str:
        if s and not _UNSAFE.search(s):
            return s

        # Paths cannot contain a '"' on Windows, so we don't need to worry
        # about nuanced counting here.
        return f'"{s}\\"' if s.endswith("\\") else f'"{s}"'
else:

    def _quote(s: str) -> str:
        return s


def _get_config_var(name: str, fmt: str = "{}", quote: bool = False) -> list[str]:
    var = sysconfig.get_config_var(name)
    if not var:
        return []
    flag = fmt.format(str(var).strip())
    return [_quote(flag) if quote else flag]


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
    Unix-style command-line compiler (GCC/Clang). Based on python-config.
    """
    flags = [_quote(f"-I{d}") for d in get_include_dirs()]
    # CFLAGS/LDFLAGS/LIBS/SYSLIBS are pre-composed multi-flag strings, passed
    # through as-is (like python-config does)
    flags += _get_config_var("CFLAGS")
    flags.append("-std=c++17")
    return " ".join(flags)


def get_ldflags(embed: bool = False) -> str:
    """
    Return the link flags for building a simple extension (or, with
    embed=True, an embedding program) with a Unix-style command-line
    compiler (GCC/Clang). Based on python-config.
    """
    flags = _get_config_var("LDFLAGS")

    if embed:
        flags += _get_config_var("LIBDIR", "-L{}", quote=True)
        if not sysconfig.get_config_var("Py_ENABLE_SHARED"):
            flags += _get_config_var("LIBPL", "-L{}", quote=True)
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
