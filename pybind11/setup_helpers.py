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
from distutils.command.build_ext import build_ext


# It is recommended to use PEP 518 builds if using this module. However, this
# file explicitly supports being copied into a user's project directory
# standalone, and pulling pybind11 with the deprecated setup_requires feature.


class DelayedPybindInclude(object):
    """
    Helper class to determine the pybind11 include path The purpose of this
    class is to postpone importing pybind11 until it is actually installed, so
    that the ``get_include()`` method can be invoked if pybind11 is loaded via
    setup_requires.
    """

    def __str__(self):
        import pybind11

        return pybind11.get_include()


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


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """
    Return a boolean indicating whether a flag name is supported on the
    specified compiler.
    """

    with tmp_chdir():
        fname = "flagcheck.cpp"
        with open(fname, "w") as f:
            f.write("int main (int argc, char **argv) { return 0; }")

        try:
            compiler.compile([fname], extra_postargs=[flagname])
            return True
        except distutils.errors.CompileError:
            return False


def cpp_flag(compiler, value=None):
    """
    Return the ``-std=c++[11/14/17]`` compiler flag.
    The newer version is preferred over c++11 (when it is available).
    """

    flags = ["-std=c++17", "-std=c++14", "-std=c++11"]

    if value is not None:
        mapping = {17: 0, 14: 1, 11: 2}
        flags = [flags[mapping[value]]]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError("Unsupported compiler -- at least C++11 support is needed!")


c_opts = {
    "msvc": ["/EHsc"],
    "unix": [],
}

l_opts = {
    "msvc": [],
    "unix": [],
}

if sys.platform == "darwin":
    darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.9"]
    c_opts["unix"] += darwin_opts
    l_opts["unix"] += darwin_opts


class BuildExt(build_ext):
    """
    Customized build_ext that can be further customized by users.

    Most use cases can be addressed by adding items to the extensions.
    However, if you need to customize, try:

        class BuildExt(pybind11.setup_utils.BuildExt):
            def build_extensions(self):
                # Do something here, like add things to extensions

            super(BuildExt, self).build_extensions()

    One simple customization point is provided: ``self.cxx_std`` lets
    you set a C++ standard (None is the default search).
    """

    def build_extensions(self):
        ct = self.compiler.compiler_type
        comp_opts = c_opts.get(ct, [])
        link_opts = l_opts.get(ct, [])
        if ct == "unix":
            comp_opts.append(cpp_flag(self.compiler, getattr(self, "cxx_std", None)))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                comp_opts.append("-fvisibility=hidden")

        for ext in self.extensions:
            ext.extra_compile_args += comp_opts
            ext.extra_link_args += link_opts
            ext.include_dirs += [DelayedPybindInclude()]

        # Python 2 doesn't allow super here, since it's not a "class"
        build_ext.build_extensions(self)
