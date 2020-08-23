#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Setup script for PyPI; use CMakeFile.txt to build extension modules

import contextlib
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from distutils.command.install_headers import install_headers

from setuptools import setup

# For now, there are three parts to this package. Besides the "normal" module:
# PYBIND11_USE_HEADERS will include the python-headers files.
# PYBIND11_USE_SYSTEM will include the sys.prefix files (CMake and headers).
# The final version will likely only include the normal module or come in
# different versions.

use_headers = os.environ.get("PYBIND11_USE_HEADERS", False)
use_system = os.environ.get("PYBIND11_USE_SYSTEM", False)

setup_opts = dict()

# In a PEP 518 build, this will be in its own environment, so it will not
# create extra files in the source

DIR = os.path.abspath(os.path.dirname(__file__))

prexist_include = os.path.exists("pybind11/include")
prexist_share = os.path.exists("pybind11/share")


@contextlib.contextmanager
def monkey_patch_file(input_file):
    "Allow a file to be temporarily modified"

    with open(os.path.join(DIR, input_file), "r") as f:
        contents = f.read()
    try:
        yield contents
    finally:
        with open(os.path.join(DIR, input_file), "w") as f:
            f.write(contents)


@contextlib.contextmanager
def TemporaryDirectory():  # noqa: N802
    "Prepare a temporary directory, cleanup when done"
    try:
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


@contextlib.contextmanager
def remove_output(*sources):
    try:
        yield
    finally:
        for src in sources:
            shutil.rmtree(src)


def check_compare(input_set, *patterns):
    "Just a quick way to make sure all files are present"
    disk_files = set()
    for pattern in patterns:
        disk_files |= set(glob.glob(pattern, recursive=True))

    assert input_set == disk_files, "{} setup.py only, {} on disk only".format(
        input_set - disk_files, disk_files - input_set
    )


class InstallHeadersNested(install_headers):
    def run(self):
        headers = self.distribution.headers or []
        for header in headers:
            # Remove include/*/
            short_header = header.split("/", 2)[-1]

            dst = os.path.join(self.install_dir, os.path.dirname(short_header))
            self.mkpath(dst)
            (out, _) = self.copy_file(header, dst)
            self.outfiles.append(out)


main_headers = {
    "include/pybind11/attr.h",
    "include/pybind11/buffer_info.h",
    "include/pybind11/cast.h",
    "include/pybind11/chrono.h",
    "include/pybind11/common.h",
    "include/pybind11/complex.h",
    "include/pybind11/eigen.h",
    "include/pybind11/embed.h",
    "include/pybind11/eval.h",
    "include/pybind11/functional.h",
    "include/pybind11/iostream.h",
    "include/pybind11/numpy.h",
    "include/pybind11/operators.h",
    "include/pybind11/options.h",
    "include/pybind11/pybind11.h",
    "include/pybind11/pytypes.h",
    "include/pybind11/stl.h",
    "include/pybind11/stl_bind.h",
}

detail_headers = {
    "include/pybind11/detail/class.h",
    "include/pybind11/detail/common.h",
    "include/pybind11/detail/descr.h",
    "include/pybind11/detail/init.h",
    "include/pybind11/detail/internals.h",
    "include/pybind11/detail/typeid.h",
}

headers = main_headers | detail_headers
check_compare(headers, "include/**/*.h")

if use_headers:
    setup_opts["headers"] = headers
    setup_opts["cmdclass"] = {"install_headers": InstallHeadersNested}

cmake_files = {
    "pybind11/share/cmake/pybind11/FindPythonLibsNew.cmake",
    "pybind11/share/cmake/pybind11/pybind11Common.cmake",
    "pybind11/share/cmake/pybind11/pybind11Config.cmake",
    "pybind11/share/cmake/pybind11/pybind11ConfigVersion.cmake",
    "pybind11/share/cmake/pybind11/pybind11NewTools.cmake",
    "pybind11/share/cmake/pybind11/pybind11Targets.cmake",
    "pybind11/share/cmake/pybind11/pybind11Tools.cmake",
}


package_headers = set("pybind11/{}".format(h) for h in headers)
package_files = package_headers | cmake_files

# Generate the files if they are not generated (will be present in tarball)
GENERATED = (
    []
    if all(os.path.exists(h) for h in package_files)
    else ["pybind11/include", "pybind11/share"]
)
with remove_output(*GENERATED):
    # Generate the files if they are not present.
    if GENERATED:
        with TemporaryDirectory() as tmpdir:
            cmd = ["cmake", "-S", ".", "-B", tmpdir] + [
                "-DCMAKE_INSTALL_PREFIX=pybind11",
                "-DBUILD_TESTING=OFF",
                "-DPYBIND11_NOPYTHON=ON",
            ]
            cmake_opts = dict(cwd=DIR, stdout=sys.stdout, stderr=sys.stderr)
            subprocess.check_call(cmd, **cmake_opts)
            subprocess.check_call(["cmake", "--install", tmpdir], **cmake_opts)

    # Make sure all files are present
    check_compare(package_files, "pybind11/include/**/*.h", "pybind11/share/**/*.cmake")

    if use_system:
        setup_opts["data_files"] = [
            ("share/cmake", cmake_files),
            ("include/pybind11", main_headers),
            ("include/pybind11/detail", detail_headers),
        ]

    # Remove the cmake / ninja requirements as now all files are guaranteed to exist
    if GENERATED:
        REQUIRES = re.compile(r"requires\s*=.+?\]", re.DOTALL | re.MULTILINE)
        with monkey_patch_file("pyproject.toml") as txt:
            with open("pyproject.toml", "w") as f:
                new_txt = REQUIRES.sub('requires = ["setuptools", "wheel"]', txt)
                f.write(new_txt)

            setup(**setup_opts)
    else:
        setup(**setup_opts)
