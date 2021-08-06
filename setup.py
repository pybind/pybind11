#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Setup script for PyPI; use CMakeFile.txt to build extension modules

import contextlib
import os
import re
import shutil
import string
import subprocess
import sys
import tempfile
import io

import setuptools.command.sdist

DIR = os.path.abspath(os.path.dirname(__file__))
VERSION_REGEX = re.compile(
    r"^\s*#\s*define\s+PYBIND11_VERSION_([A-Z]+)\s+(.*)$", re.MULTILINE
)


def build_expected_version_hex(matches):
    patch_level_serial = matches["PATCH"]
    serial = None
    try:
        major = int(matches["MAJOR"])
        minor = int(matches["MINOR"])
        flds = patch_level_serial.split(".")
        if flds:
            patch = int(flds[0])
            level = None
            if len(flds) == 1:
                level = "0"
                serial = 0
            elif len(flds) == 2:
                level_serial = flds[1]
                for level in ("a", "b", "c", "dev"):
                    if level_serial.startswith(level):
                        serial = int(level_serial[len(level) :])
                        break
    except ValueError:
        pass
    if serial is None:
        msg = 'Invalid PYBIND11_VERSION_PATCH: "{}"'.format(patch_level_serial)
        raise RuntimeError(msg)
    return "0x{:02x}{:02x}{:02x}{}{:x}".format(
        major, minor, patch, level[:1].upper(), serial
    )


# PYBIND11_GLOBAL_SDIST will build a different sdist, with the python-headers
# files, and the sys.prefix files (CMake and headers).

global_sdist = os.environ.get("PYBIND11_GLOBAL_SDIST", False)

setup_py = "tools/setup_global.py.in" if global_sdist else "tools/setup_main.py.in"
extra_cmd = 'cmdclass["sdist"] = SDist\n'

to_src = (
    ("pyproject.toml", "tools/pyproject.toml"),
    ("setup.py", setup_py),
)

# Read the listed version
with open("pybind11/_version.py") as f:
    code = compile(f.read(), "pybind11/_version.py", "exec")
loc = {}
exec(code, loc)
version = loc["__version__"]

# Verify that the version matches the one in C++
with io.open("include/pybind11/detail/common.h", encoding="utf8") as f:
    matches = dict(VERSION_REGEX.findall(f.read()))
cpp_version = "{MAJOR}.{MINOR}.{PATCH}".format(**matches)
if version != cpp_version:
    msg = "Python version {} does not match C++ version {}!".format(
        version, cpp_version
    )
    raise RuntimeError(msg)

version_hex = matches.get("HEX", "MISSING")
expected_version_hex = build_expected_version_hex(matches)
if version_hex != expected_version_hex:
    msg = "PYBIND11_VERSION_HEX {} does not match expected value {}!".format(
        version_hex,
        expected_version_hex,
    )
    raise RuntimeError(msg)


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
