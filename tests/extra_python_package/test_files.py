from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Generator

# These tests must be run explicitly

DIR = Path(__file__).parent.resolve()
MAIN_DIR = DIR.parent.parent


# Newer pytest has global path setting, but keeping old pytest for now
sys.path.append(str(MAIN_DIR / "tools"))

from make_global import get_global  # noqa: E402

HAS_UV = shutil.which("uv") is not None
UV_ARGS = ["--installer=uv"] if HAS_UV else []

PKGCONFIG = """\
prefix=${{pcfiledir}}/../../
includedir=${{prefix}}/include

Name: pybind11
Description: Seamless operability between C++11 and Python
Version: {VERSION}
Cflags: -I${{includedir}}
"""


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
    "include/pybind11/gil.h",
    "include/pybind11/gil_safe_call_once.h",
    "include/pybind11/gil_simple.h",
    "include/pybind11/iostream.h",
    "include/pybind11/native_enum.h",
    "include/pybind11/numpy.h",
    "include/pybind11/operators.h",
    "include/pybind11/options.h",
    "include/pybind11/pybind11.h",
    "include/pybind11/pytypes.h",
    "include/pybind11/stl.h",
    "include/pybind11/stl_bind.h",
    "include/pybind11/trampoline_self_life_support.h",
    "include/pybind11/type_caster_pyobject_ptr.h",
    "include/pybind11/typing.h",
    "include/pybind11/warnings.h",
}

conduit_headers = {
    "include/pybind11/conduit/README.txt",
    "include/pybind11/conduit/pybind11_conduit_v1.h",
    "include/pybind11/conduit/pybind11_platform_abi_id.h",
    "include/pybind11/conduit/wrap_include_python_h.h",
}

detail_headers = {
    "include/pybind11/detail/class.h",
    "include/pybind11/detail/common.h",
    "include/pybind11/detail/cpp_conduit.h",
    "include/pybind11/detail/descr.h",
    "include/pybind11/detail/dynamic_raw_ptr_cast_if_possible.h",
    "include/pybind11/detail/function_record_pyobject.h",
    "include/pybind11/detail/init.h",
    "include/pybind11/detail/internals.h",
    "include/pybind11/detail/native_enum_data.h",
    "include/pybind11/detail/pybind11_namespace_macros.h",
    "include/pybind11/detail/struct_smart_holder.h",
    "include/pybind11/detail/type_caster_base.h",
    "include/pybind11/detail/typeid.h",
    "include/pybind11/detail/using_smart_holder.h",
    "include/pybind11/detail/value_and_holder.h",
    "include/pybind11/detail/exception_translation.h",
}

eigen_headers = {
    "include/pybind11/eigen/common.h",
    "include/pybind11/eigen/matrix.h",
    "include/pybind11/eigen/tensor.h",
}

stl_headers = {
    "include/pybind11/stl/filesystem.h",
}

cmake_files = {
    "share/cmake/pybind11/FindPythonLibsNew.cmake",
    "share/cmake/pybind11/pybind11Common.cmake",
    "share/cmake/pybind11/pybind11Config.cmake",
    "share/cmake/pybind11/pybind11ConfigVersion.cmake",
    "share/cmake/pybind11/pybind11GuessPythonExtSuffix.cmake",
    "share/cmake/pybind11/pybind11NewTools.cmake",
    "share/cmake/pybind11/pybind11Targets.cmake",
    "share/cmake/pybind11/pybind11Tools.cmake",
}

pkgconfig_files = {
    "share/pkgconfig/pybind11.pc",
}

py_files = {
    "__init__.py",
    "__main__.py",
    "_version.py",
    "commands.py",
    "py.typed",
    "setup_helpers.py",
    "share/__init__.py",
    "share/pkgconfig/__init__.py",
}

headers = main_headers | conduit_headers | detail_headers | eigen_headers | stl_headers
generated_files = cmake_files | pkgconfig_files
all_files = headers | generated_files | py_files

sdist_files = {
    "pyproject.toml",
    "LICENSE",
    "README.rst",
    "PKG-INFO",
    "SECURITY.md",
}


@contextlib.contextmanager
def preserve_file(filename: Path) -> Generator[str, None, None]:
    old_stat = filename.stat()
    old_file = filename.read_text(encoding="utf-8")
    try:
        yield old_file
    finally:
        filename.write_text(old_file, encoding="utf-8")
        os.utime(filename, (old_stat.st_atime, old_stat.st_mtime))


@contextlib.contextmanager
def build_global() -> Generator[None, None, None]:
    """
    Build global SDist and wheel.
    """

    pyproject = MAIN_DIR / "pyproject.toml"
    with preserve_file(pyproject):
        newer_txt = get_global()
        pyproject.write_text(newer_txt, encoding="utf-8")
        yield


def read_tz_file(tar: tarfile.TarFile, name: str) -> bytes:
    start = tar.getnames()[0].split("/")[0] + "/"
    inner_file = tar.extractfile(tar.getmember(f"{start}{name}"))
    assert inner_file
    with contextlib.closing(inner_file) as f:
        return f.read()


def normalize_line_endings(value: bytes) -> bytes:
    return value.replace(os.linesep.encode("utf-8"), b"\n")


def test_build_sdist(monkeypatch, tmpdir):
    monkeypatch.chdir(MAIN_DIR)

    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", f"--outdir={tmpdir}", *UV_ARGS],
        check=True,
    )

    (sdist,) = tmpdir.visit("*.tar.gz")
    version = sdist.basename.split("-")[1][:-7]

    with tarfile.open(str(sdist), "r:gz") as tar:
        simpler = {n.split("/", 1)[-1] for n in tar.getnames()[1:]}
        (pkg_info_path,) = (n for n in simpler if n.endswith("PKG-INFO"))

        pyproject_toml = read_tz_file(tar, "pyproject.toml")
        pkg_info = read_tz_file(tar, pkg_info_path).decode("utf-8")

    files = headers | sdist_files
    assert files <= simpler

    assert b'name = "pybind11"' in pyproject_toml
    assert "License-Expression: BSD-3-Clause" in pkg_info
    assert "License-File: LICENSE" in pkg_info
    assert "Provides-Extra: global" in pkg_info
    assert f'Requires-Dist: pybind11-global=={version}; extra == "global"' in pkg_info


def test_build_global_dist(monkeypatch, tmpdir):
    monkeypatch.chdir(MAIN_DIR)
    with build_global():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--outdir",
                str(tmpdir),
                *UV_ARGS,
            ],
            check=True,
        )

    (sdist,) = tmpdir.visit("*.tar.gz")

    with tarfile.open(str(sdist), "r:gz") as tar:
        simpler = {n.split("/", 1)[-1] for n in tar.getnames()[1:]}
        (pkg_info_path,) = (n for n in simpler if n.endswith("PKG-INFO"))

        pyproject_toml = read_tz_file(tar, "pyproject.toml")
        pkg_info = read_tz_file(tar, pkg_info_path).decode("utf-8")

    files = headers | sdist_files
    assert files <= simpler

    assert b'name = "pybind11-global"' in pyproject_toml
    assert "License-Expression: BSD-3-Clause" in pkg_info
    assert "License-File: LICENSE" in pkg_info
    assert "Provides-Extra: global" not in pkg_info
    assert 'Requires-Dist: pybind11-global; extra == "global"' not in pkg_info


def tests_build_wheel(monkeypatch, tmpdir):
    monkeypatch.chdir(MAIN_DIR)

    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmpdir), *UV_ARGS],
        check=True,
    )

    (wheel,) = tmpdir.visit("*.whl")

    files = {f"pybind11/{n}" for n in all_files}
    files |= {
        "dist-info/licenses/LICENSE",
        "dist-info/METADATA",
        "dist-info/RECORD",
        "dist-info/WHEEL",
        "dist-info/entry_points.txt",
    }

    with zipfile.ZipFile(str(wheel)) as z:
        names = z.namelist()
        share = zipfile.Path(z, "pybind11/share")
        pkgconfig = (share / "pkgconfig/pybind11.pc").read_text(encoding="utf-8")
        cmakeconfig = (share / "cmake/pybind11/pybind11Config.cmake").read_text(
            encoding="utf-8"
        )
        (pkg_info_path,) = (n for n in names if n.endswith("METADATA"))
        pkg_info = zipfile.Path(z, pkg_info_path).read_text(encoding="utf-8")

    trimmed = {n for n in names if "dist-info" not in n}
    trimmed |= {f"dist-info/{n.split('/', 1)[-1]}" for n in names if "dist-info" in n}

    assert files == trimmed

    assert 'set(pybind11_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")' in cmakeconfig

    version = wheel.basename.split("-")[1]
    simple_version = ".".join(version.split(".")[:3])
    pkgconfig_expected = PKGCONFIG.format(VERSION=simple_version)
    assert pkgconfig_expected == pkgconfig

    assert "License-Expression: BSD-3-Clause" in pkg_info
    assert "License-File: LICENSE" in pkg_info
    assert "Provides-Extra: global" in pkg_info
    assert f'Requires-Dist: pybind11-global=={version}; extra == "global"' in pkg_info


def tests_build_global_wheel(monkeypatch, tmpdir):
    monkeypatch.chdir(MAIN_DIR)
    with build_global():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(tmpdir),
                *UV_ARGS,
            ],
            check=True,
        )

    (wheel,) = tmpdir.visit("*.whl")

    files = {f"data/data/{n}" for n in headers}
    files |= {f"data/headers/{n[8:]}" for n in headers}
    files |= {f"data/data/{n}" for n in generated_files}
    files |= {
        "dist-info/licenses/LICENSE",
        "dist-info/METADATA",
        "dist-info/WHEEL",
        "dist-info/RECORD",
    }

    with zipfile.ZipFile(str(wheel)) as z:
        names = z.namelist()
        beginning = names[0].split("/", 1)[0].rsplit(".", 1)[0]

        share = zipfile.Path(z, f"{beginning}.data/data/share")
        pkgconfig = (share / "pkgconfig/pybind11.pc").read_text(encoding="utf-8")
        cmakeconfig = (share / "cmake/pybind11/pybind11Config.cmake").read_text(
            encoding="utf-8"
        )

        (pkg_info_path,) = (n for n in names if n.endswith("METADATA"))
        pkg_info = zipfile.Path(z, pkg_info_path).read_text(encoding="utf-8")

    assert "License-Expression: BSD-3-Clause" in pkg_info
    assert "License-File: LICENSE" in pkg_info
    assert "Provides-Extra: global" not in pkg_info
    assert 'Requires-Dist: pybind11-global; extra == "global"' not in pkg_info

    trimmed = {n[len(beginning) + 1 :] for n in names}

    assert files == trimmed

    assert 'set(pybind11_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")' in cmakeconfig

    version = wheel.basename.split("-")[1]
    simple_version = ".".join(version.split(".")[:3])
    pkgconfig_expected = PKGCONFIG.format(VERSION=simple_version)
    assert pkgconfig_expected == pkgconfig
