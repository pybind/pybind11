# -*- coding: utf-8 -*-

"""
This module provides helpers for C++11+ projects using pybind11.

LICENSE:

Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import contextlib
import os
import shutil
import sys
import tempfile
import threading
import warnings

try:
    from setuptools.command.build_ext import build_ext as _build_ext
    from setuptools import Extension as _Extension
except ImportError:
    from distutils.command.build_ext import build_ext as _build_ext
    from distutils.extension import Extension as _Extension

import distutils.errors


WIN = sys.platform.startswith("win32")
PY2 = sys.version_info[0] < 3
MACOS = sys.platform.startswith("darwin")
STD_TMPL = "/std:c++{}" if WIN else "-std=c++{}"


# It is recommended to use PEP 518 builds if using this module. However, this
# file explicitly supports being copied into a user's project directory
# standalone, and pulling pybind11 with the deprecated setup_requires feature.
# If you copy the file, remember to add it to your MANIFEST.in, and add the current
# directory into your path if it sits beside your setup.py.


class Pybind11Extension(_Extension):
    """
    Build a C++11+ Extension module with pybind11. This automatically adds the
    recommended flags when you init the extension and assumes C++ sources - you
    can further modify the options yourself.

    The customizations are:

    * ``/EHsc`` and ``/bigobj`` on Windows
    * ``stdlib=libc++`` on macOS
    * ``visibility=hidden`` and ``-g0`` on Unix

    Finally, you can set ``cxx_std`` via constructor or afterwords to enable
    flags for C++ std, and a few extra helper flags related to the C++ standard
    level. It is _highly_ recommended you either set this, or use the provided
    ``build_ext``, which will search for the highest supported extension for
    you if the ``cxx_std`` property is not set. Do not set the ``cxx_std``
    property more than once, as flags are added when you set it. Set the
    property to None to disable the addition of C++ standard flags.

    If you want to add pybind11 headers manually, for example for an exact
    git checkout, then set ``include_pybind11=False``.

    Warning: do not use property-based access to the instance on Python 2 -
    this is an ugly old-style class due to Distutils.
    """

    def _add_cflags(self, *flags):
        for flag in flags:
            if flag not in self.extra_compile_args:
                self.extra_compile_args.append(flag)

    def _add_lflags(self, *flags):
        for flag in flags:
            if flag not in self.extra_compile_args:
                self.extra_link_args.append(flag)

    def __init__(self, *args, **kwargs):

        self._cxx_level = 0
        cxx_std = kwargs.pop("cxx_std", 0)

        if "language" not in kwargs:
            kwargs["language"] = "c++"

        include_pybind11 = kwargs.pop("include_pybind11", True)

        # Can't use super here because distutils has old-style classes in
        # Python 2!
        _Extension.__init__(self, *args, **kwargs)

        # Include the installed package pybind11 headers
        if include_pybind11:
            # If using setup_requires, this fails the first time - that's okay
            try:
                import pybind11

                pyinc = pybind11.get_include()

                if pyinc not in self.include_dirs:
                    self.include_dirs.append(pyinc)
            except ImportError:
                pass

        # Have to use the accessor manually to support Python 2 distutils
        Pybind11Extension.cxx_std.__set__(self, cxx_std)

        if WIN:
            self._add_cflags("/EHsc", "/bigobj")
        else:
            self._add_cflags("-fvisibility=hidden", "-g0")
            if MACOS:
                self._add_cflags("-stdlib=libc++")
                self._add_lflags("-stdlib=libc++")

    @property
    def cxx_std(self):
        """
        The CXX standard level. If set, will add the required flags. If left
        at 0, it will trigger an automatic search when pybind11's build_ext
        is used. If None, will have no effect.  Besides just the flags, this
        may add a register warning/error fix for Python 2 or macos-min 10.9
        or 10.14.
        """
        return self._cxx_level

    @cxx_std.setter
    def cxx_std(self, level):

        if self._cxx_level:
            warnings.warn("You cannot safely change the cxx_level after setting it!")

        # MSVC 2015 Update 3 and later only have 14 (and later 17) modes
        if WIN and level == 11:
            level = 14

        self._cxx_level = level

        if not level:
            return

        self.extra_compile_args.append(STD_TMPL.format(level))

        if MACOS and "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
            # C++17 requires a higher min version of macOS
            macosx_min = "-mmacosx-version-min=" + ("10.9" if level < 17 else "10.14")
            self.extra_compile_args.append(macosx_min)
            self.extra_link_args.append(macosx_min)

        if PY2:
            if level >= 17:
                self.extra_compile_args.append("/wd503" if WIN else "-Wno-register")
            elif not WIN and level >= 14:
                self.extra_compile_args.append("-Wno-deprecated-register")


# Just in case someone clever tries to multithread
tmp_chdir_lock = threading.Lock()
cpp_cache_lock = threading.Lock()


@contextlib.contextmanager
def tmp_chdir():
    "Prepare and enter a temporary directory, cleanup when done"

    # Threadsafe
    with tmp_chdir_lock:
        olddir = os.getcwd()
        try:
            tmpdir = tempfile.mkdtemp()
            os.chdir(tmpdir)
            yield tmpdir
        finally:
            os.chdir(olddir)
            shutil.rmtree(tmpdir)


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flag):
    """
    Return the flag if a flag name is supported on the
    specified compiler, otherwise None (can be used as a boolean).
    If multiple flags are passed, return the first that matches.
    """

    with tmp_chdir():
        fname = "flagcheck.cpp"
        with open(fname, "w") as f:
            f.write("int main (int argc, char **argv) { return 0; }")

        try:
            compiler.compile([fname], extra_postargs=[flag])
        except distutils.errors.CompileError:
            return False
        return True


# Every call will cache the result
cpp_flag_cache = None


def auto_cpp_level(compiler):
    """
    Return the max supported C++ std level (17, 14, or 11).
    """

    global cpp_flag_cache

    # If this has been previously calculated with the same args, return that
    with cpp_cache_lock:
        if cpp_flag_cache:
            return cpp_flag_cache

    levels = [17, 14] + ([] if WIN else [11])

    for level in levels:
        if has_flag(compiler, STD_TMPL.format(level)):
            with cpp_cache_lock:
                cpp_flag_cache = level
            return level

    msg = "Unsupported compiler -- at least C++11 support is needed!"
    raise RuntimeError(msg)


class build_ext(_build_ext):  # noqa: N801
    """
    Customized build_ext that allows an auto-search for the highest supported
    C++ level for Pybind11Extension.
    """

    def build_extensions(self):
        """
        Build extensions, injecting C++ std for Pybind11Extension if needed.
        """

        for ext in self.extensions:
            if hasattr(ext, "_cxx_level") and ext._cxx_level == 0:
                # Python 2 syntax - old-style distutils class
                ext.__class__.cxx_std.__set__(ext, auto_cpp_level(self.compiler))

        # Python 2 doesn't allow super here, since distutils uses old-style
        # classes!
        _build_ext.build_extensions(self)
