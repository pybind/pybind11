#!/usr/bin/env python

# Setup script for PyPI; use CMakeFile.txt to build extension modules

import contextlib
import os
import re
import shutil
import string
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import setuptools.command.sdist

DIR = Path(__file__).parent.absolute()
VERSION_REGEX = re.compile(
    r"^\s*#\s*define\s+PYBIND11_VERSION_([A-Z]+)\s+(.*)$", re.MULTILINE
)
VERSION_FILE = Path("pybind11/_version.py")
COMMON_FILE = Path("include/pybind11/detail/common.h")


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
    return (
        "0x"
        + "{:02x}{:02x}{:02x}{}{:x}".format(
            major, minor, patch, level[:1], serial
        ).upper()
    )


# PYBIND11_GLOBAL_SDIST will build a different sdist, with the python-headers
# files, and the sys.prefix files (CMake and headers).

global_sdist = os.environ.get("PYBIND11_GLOBAL_SDIST", False)

setup_py = Path(
    "tools/setup_global.py.in" if global_sdist else "tools/setup_main.py.in"
)
extra_cmd = 'cmdclass["sdist"] = SDist\n'

to_src = (
    (Path("pyproject.toml"), Path("tools/pyproject.toml")),
    (Path("setup.py"), setup_py),
)


# Read the listed version
loc = {}
code = compile(VERSION_FILE.read_text(encoding="utf-8"), "pybind11/_version.py", "exec")
exec(code, loc)
version = loc["__version__"]

# Verify that the version matches the one in C++
matches = dict(VERSION_REGEX.findall(COMMON_FILE.read_text(encoding="utf8")))
cpp_version = "{MAJOR}.{MINOR}.{PATCH}".format(**matches)
if version != cpp_version:
    msg = "Python version {} does not match C++ version {}!".format(
        version, cpp_version
    )
    raise RuntimeError(msg)

version_hex = matches.get("HEX", "MISSING")
exp_version_hex = build_expected_version_hex(matches)
if version_hex != exp_version_hex:
    msg = "PYBIND11_VERSION_HEX {} does not match expected value {}!".format(
        version_hex, exp_version_hex
    )
    raise RuntimeError(msg)


def get_and_replace(filename: Path, binary: bool = False, **opts: str):
    if binary:
        contents = filename.read_bytes()
        return string.Template(contents.decode()).substitute(opts).encode()

    contents = filename.read_text()
    return string.Template(contents).substitute(opts)


# Use our input files instead when making the SDist (and anything that depends
# on it, like a wheel)
class SDist(setuptools.command.sdist.sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)

        for to, src in to_src:
            txt = get_and_replace(src, binary=True, version=version, extra_cmd="")

            dest = Path(base_dir) / to

            # This is normally linked, so unlink before writing!
            dest.unlink()
            dest.write_bytes(txt)


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
        if "CMAKE_ARGS" in os.environ:
            fcommand = [
                c
                for c in os.environ["CMAKE_ARGS"].split()
                if "DCMAKE_INSTALL_PREFIX" not in c
            ]
            cmd += fcommand
        cmake_opts = dict(cwd=DIR, stdout=sys.stdout, stderr=sys.stderr)
        subprocess.run(cmd, check=True, **cmake_opts)
        subprocess.run(["cmake", "--install", tmpdir], check=True, **cmake_opts)

    txt = get_and_replace(setup_py, version=version, extra_cmd=extra_cmd)
    code = compile(txt, setup_py, "exec")
    exec(code, {"SDist": SDist})
