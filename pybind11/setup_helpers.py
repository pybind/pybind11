# -*- coding: utf-8 -*-

"""
This module provides a way to check to see if flag is available,
has_flag (built-in to distutils.CCompiler in Python 3.6+), and
a cpp_flag function, which will compute the highest available
flag (or if a flag is supported).

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

import distutils.errors
import distutils.command.build_ext


WIN = sys.platform.startswith("win")
PY2 = sys.version_info[0] < 3


# It is recommended to use PEP 518 builds if using this module. However, this
# file explicitly supports being copied into a user's project directory
# standalone, and pulling pybind11 with the deprecated setup_requires feature.
# If you copy the file, remember to add it to your MANIFEST.in, and add the current
# directory into your path if it sits beside your setup.py.


# Just in case someone clever tries to multithread
tmp_chdir_lock = threading.Lock()


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


class build_ext(distutils.command.build_ext.build_ext):  # noqa: N801
    """
    Customized build_ext that can be further customized by users.

    Most use cases can be addressed by adding items to the extensions.
    However, if you need to customize beyond the customization points provided,
    try:

        class build_ext(pybind11.setup_utils.build_ext):
            def build_extensions(self):
                # Do something here, like add things to extensions

            # super only works on Python 3 due to distutils oddities
            pybind11.setup_utils.build_ext.build_extensions(self)

    Simple customization points are provided: CPP_FLAGS, UNIX_FLAGS,
    LINUX_FLAGS, DARWIN_FLAGS, and WIN_FLAGS (along with ``*_LINK_FLAGS``
    versions). You can override these per class or per instance.

    If CPP_FLAGS is empty, no search is done. The first supported flag is used.
    Reasonable defaults are selected for the flags for optimal extensions. An
    ALL_COMPILERS set lists flags that will always pass "has_flag".

    Two flags are special, and are not listed here. One is the Python 2 +
    C++14+/C++17 register flag; this is added by cpp_flags. The other is the
    macos-version-min flag, which is only added if MACOSX_DEPLOYMENT_TARGET is
    not set, and is based on the C++ standard selected.
    """

    CPP_FLAGS = ("-std=c++17", "-std=c++14", "-std=c++11")
    LINUX_FLAGS = ()
    LINUX_LINK_FLAGS = ()
    UNIX_FLAGS = ("-fvisibility=hidden", "-g0")
    UNIX_LINK_FLAGS = ()
    DARWIN_FLAGS = ("-stdlib=libc++",)
    DARWIN_LINK_FLAGS = ("-stdlib=libc++",)
    WIN_FLAGS = ("/EHsc", "/bigobj")
    WIN_LINK_FLAGS = ()
    ALL_COMPILERS = {"-g0", "/EHsc", "/bigobj"}

    # cf http://bugs.python.org/issue26689
    def has_flag(self, *flagnames):
        """
        Return the flag if a flag name is supported on the
        specified compiler, otherwise None (can be used as a boolean).
        If multiple flags are passed, return the first that matches.
        """

        with tmp_chdir():
            fname = "flagcheck.cpp"
            for flagname in flagnames:
                if flagname in self.ALL_COMPILERS or "mmacosx-version-min" in flagname:
                    return flagname
                with open(fname, "w") as f:
                    f.write("int main (int argc, char **argv) { return 0; }")

                try:
                    self.compiler.compile([fname], extra_postargs=[flagname])
                    return flagname
                except distutils.errors.CompileError:
                    pass

        return None

    def cpp_flags(self):
        """
        Return the ``-std=c++[11/14/17]`` compiler flag(s) as a list. May add
        register fix for Python 2.  The newer version is preferred over c++11
        (when it is available). Windows will not fail, since MSVC 15 doesn't
        have these flags but supports C++14 (for the most part). The first flag
        is always the C++ selection flag.
        """

        # None or missing attribute, provide default list; if empty list, return nothing
        if not self.CPP_FLAGS:
            return []

        # Windows uses a different form
        if WIN:
            # MSVC 2017+
            flags = [
                "/std:{}".format(flag[5:]).replace("11", "14")
                for flag in self.CPP_FLAGS
            ]
        else:
            flags = self.CPP_FLAGS

        flag = self.has_flag(*flags)

        if flag is None:
            # On Windows, the default is to support C++14 on MSVC 2015, and it is
            # not a specific flag so not failing here if on Windows. An empty list
            # also passes since it doesn't look for anything.
            if WIN:
                return []
            else:
                msg = "Unsupported compiler -- at least C++11 support is needed!"
                raise RuntimeError(msg)

        if PY2:
            try:
                value = int(flag[-2:])
                if value >= 17:
                    return [flag, "/wd503" if WIN else "-Wno-register"]
                elif not WIN and value >= 14:
                    return [flag, "-Wno-deprecated-register"]
            except ValueError:
                return [flag, "/wd503" if WIN else "-Wno-register"]

        return [flag]

    def build_extensions(self):
        """
        Build extensions, injecting extra flags and includes as needed.
        """
        # Import here to support `setup_requires` if someone really has to use it.
        import pybind11

        def valid_flags(flagnames):
            return [flag for flag in flagnames if self.has_flag(flag)]

        extra_compile_args = []
        extra_link_args = []

        cpp_flags = self.cpp_flags()
        extra_compile_args += cpp_flags

        if WIN:
            extra_compile_args += valid_flags(self.WIN_FLAGS)
            extra_link_args += self.WIN_LINK_FLAGS
        else:
            extra_compile_args += valid_flags(self.UNIX_FLAGS)
            extra_link_args += self.UNIX_LINK_FLAGS

            if sys.platform.startswith("darwin"):
                extra_compile_args += valid_flags(self.DARWIN_FLAGS)
                extra_link_args += self.DARWIN_LINK_FLAGS

                if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
                    macosx_min = "-mmacosx-version-min=10.9"

                    # C++17 requires a higher min version of macOS
                    if cpp_flags:
                        try:
                            if int(cpp_flags[0][-2:]) >= 17:
                                macosx_min = "-mmacosx-version-min=10.14"
                        except ValueError:
                            pass

                    extra_compile_args.append(macosx_min)
                    extra_link_args.append(macosx_min)

            else:
                extra_compile_args += valid_flags(self.LINUX_FLAGS)
                extra_link_args += self.LINUX_LINK_FLAGS

        for ext in self.extensions:
            ext.extra_compile_args += extra_compile_args
            ext.extra_link_args += extra_link_args
            ext.include_dirs += [pybind11.get_include()]

        # Python 2 doesn't allow super here, since it's not a "class"
        distutils.command.build_ext.build_ext.build_extensions(self)
