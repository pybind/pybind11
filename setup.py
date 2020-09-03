#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Setup script for PyPI; use CMakeFile.txt to build extension modules

import contextlib
import os
import shutil
import string
import subprocess
import sys
import tempfile

import setuptools.command.sdist

DIR = os.path.abspath(os.path.dirname(__file__))

# PYBIND11_GLOBAL_SDIST will build a different sdist, with the python-headers
# files, and the sys.prefix files (CMake and headers).

alt_sdist = os.environ.get("PYBIND11_GLOBAL_SDIST", False)

version_py = "pybind11/_version.py"
setup_py = "tools/setup_global.py.in" if alt_sdist else "tools/setup_main.py.in"
extra_cmd = 'cmdclass["sdist"] = SDist\n'

to_src = (
    (version_py, "tools/_version.py.in"),
    ("pyproject.toml", "tools/pyproject.toml"),
    ("setup.py", setup_py),
)


with open(version_py) as f:
    loc = {"__file__": version_py}
    code = compile(f.read(), version_py, "exec")
    exec(code, loc)
    version = loc["__version__"]


def get_and_replace(filename, binary=False, **opts):
    with open(filename, "rb" if binary else "r") as f:
        contents = f.read()
    # Replacement has to be done on text in Python 3 (both work in Python 2)
    if binary:
        return string.Template(contents.decode()).substitute(opts).encode()
    else:
        return string.Template(contents).substitute(opts)


# Use our input files instead when making the SDist (and anything that depends
# on it, like a wheel)
class SDist(setuptools.command.sdist.sdist):
    def make_release_tree(self, base_dir, files):
        setuptools.command.sdist.sdist.make_release_tree(self, base_dir, files)

        for to, src in to_src:
            txt = get_and_replace(src, binary=True, version=version, extra_cmd="")

            dest = os.path.join(base_dir, to)

            # This is normally linked, so unlink before writing!
            os.unlink(dest)
            with open(dest, "wb") as f:
                f.write(txt)


# Backport from Python 3
@contextlib.contextmanager
def TemporaryDirectory():  # noqa: N802
    "Prepare a temporary directory, cleanup when done"
    try:
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


# Remove the CMake install directory when done
@contextlib.contextmanager
def remove_output(*sources):
    try:
        yield
    finally:
        for src in sources:
            shutil.rmtree(src)


with remove_output("pybind11/include", "pybind11/share"):
    # Generate the files if they are not present.
    with TemporaryDirectory() as tmpdir:
        cmd = ["cmake", "-S", ".", "-B", tmpdir] + [
            "-DCMAKE_INSTALL_PREFIX=pybind11",
            "-DBUILD_TESTING=OFF",
            "-DPYBIND11_NOPYTHON=ON",
        ]
        cmake_opts = dict(cwd=DIR, stdout=sys.stdout, stderr=sys.stderr)
        subprocess.check_call(cmd, **cmake_opts)
        subprocess.check_call(["cmake", "--install", tmpdir], **cmake_opts)

    txt = get_and_replace(setup_py, version=version, extra_cmd=extra_cmd)
    code = compile(txt, setup_py, "exec")
    exec(code, {"SDist": SDist})
