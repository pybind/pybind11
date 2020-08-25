#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Setup script for PyPI; use CMakeFile.txt to build extension modules

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile

# PYBIND11_ALT_SDIST will build a different sdist, with the python-headers
# files, and the sys.prefix files (CMake and headers).

alt_sdist = os.environ.get("PYBIND11_ALT_SDIST", False)
setup_py = "tools/setup_alt.py" if alt_sdist else "tools/setup_main.py"
pyproject_toml = "tools/pyproject.toml"

# In a PEP 518 build, this will be in its own environment, so it will not
# create extra files in the source

DIR = os.path.abspath(os.path.dirname(__file__))


@contextlib.contextmanager
def monkey_patch_file(input_file, replacement_file):
    "Allow a file to be temporarily replaced"
    inp_file = os.path.abspath(os.path.join(DIR, input_file))
    rep_file = os.path.abspath(os.path.join(DIR, replacement_file))

    with open(inp_file, "rb") as f:
        contents = f.read()
    with open(rep_file, "rb") as f:
        replacement = f.read()
    try:
        with open(inp_file, "wb") as f:
            f.write(replacement)
        yield
    finally:
        with open(inp_file, "wb") as f:
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

    with monkey_patch_file("pyproject.toml", pyproject_toml):
        with monkey_patch_file("setup.py", setup_py):
            with open(setup_py) as f:
                code = compile(f.read(), setup_py, "exec")
                exec(code, globals(), {})
