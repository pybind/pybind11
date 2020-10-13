# -*- coding: utf-8 -*-
import os


DIR = os.path.abspath(os.path.dirname(__file__))


def get_include(user=False):
    # type: (bool) -> str
    installed_path = os.path.join(DIR, "include")
    source_path = os.path.join(os.path.dirname(DIR), "include")
    return installed_path if os.path.exists(installed_path) else source_path


def get_cmake_dir():
    # type: () -> str
    cmake_installed_path = os.path.join(DIR, "share", "cmake", "pybind11")
    if os.path.exists(cmake_installed_path):
        return cmake_installed_path
    else:
        msg = "pybind11 not installed, installation required to access the CMake files"
        raise ImportError(msg)
