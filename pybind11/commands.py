import os
import sys
import sysconfig

DIR = os.path.abspath(os.path.dirname(__file__))


def get_include(user: bool = False) -> str:  # pylint: disable=unused-argument
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


def get_cflags() -> str:
    """
    Gets the compile flags needed for a simple module.
    """

    flags = f"-I{sysconfig.get_path('include')} -I{sysconfig.get_path('platinclude')} -I{get_include()}"
    cflags = sysconfig.get_config_var("CFLAGS")
    if cflags:
        flags += " " + cflags
        flags += " -std=c++11"
    return flags


def get_ldflags(embed: bool) -> str:
    """
    Get the linker flags needed for a simple module.
    """

    flags = sysconfig.get_config_var("LDFLAGS") or ""

    if embed:
        if not sysconfig.get_config_var("Py_ENABLE_SHARED"):
            flags += f" -L{sysconfig.get_config_var('LIBPL') or ''}"
        pyver = sysconfig.get_config_var("VERSION")
        flags += f" -lpython{pyver}{sys.abiflags}"
        flags += " " + (sysconfig.get_config_var("LIBS") or "")
        flags += " " + (sysconfig.get_config_var("SYSLIBS") or "")
    else:
        if sys.platform.startswith("darwin"):
            flags += " -undefined dynamic_lookup"
        elif sys.platform.startswith("linux"):
            flags += " -fPIC"

    return flags


def get_extension() -> str:
    """
    Get the extension suffix on this platform
    """

    return sysconfig.get_config_var("EXT_SUFFIX") or ""
