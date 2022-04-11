import contextlib
import os
import string
import subprocess
import sys
import tarfile
import zipfile

# These tests must be run explicitly
# They require CMake 3.15+ (--install)

DIR = os.path.abspath(os.path.dirname(__file__))
MAIN_DIR = os.path.dirname(os.path.dirname(DIR))


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
    "include/pybind11/detail/type_caster_base.h",
    "include/pybind11/detail/typeid.h",
}

stl_headers = {
    "include/pybind11/stl/filesystem.h",
}

cmake_files = {
    "share/cmake/pybind11/FindPythonLibsNew.cmake",
    "share/cmake/pybind11/pybind11Common.cmake",
    "share/cmake/pybind11/pybind11Config.cmake",
    "share/cmake/pybind11/pybind11ConfigVersion.cmake",
    "share/cmake/pybind11/pybind11NewTools.cmake",
    "share/cmake/pybind11/pybind11Targets.cmake",
    "share/cmake/pybind11/pybind11Tools.cmake",
}

py_files = {
    "__init__.py",
    "__main__.py",
    "_version.py",
    "commands.py",
    "py.typed",
    "setup_helpers.py",
}

headers = main_headers | detail_headers | stl_headers
src_files = headers | cmake_files
all_files = src_files | py_files


sdist_files = {
    "pybind11",
    "pybind11/include",
    "pybind11/include/pybind11",
    "pybind11/include/pybind11/detail",
    "pybind11/include/pybind11/stl",
    "pybind11/share",
    "pybind11/share/cmake",
    "pybind11/share/cmake/pybind11",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "LICENSE",
    "MANIFEST.in",
    "README.rst",
    "PKG-INFO",
}

local_sdist_files = {
    ".egg-info",
    ".egg-info/PKG-INFO",
    ".egg-info/SOURCES.txt",
    ".egg-info/dependency_links.txt",
    ".egg-info/not-zip-safe",
    ".egg-info/top_level.txt",
}


def test_build_sdist(monkeypatch, tmpdir):

    monkeypatch.chdir(MAIN_DIR)

    out = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--outdir",
            str(tmpdir),
        ]
    )
    if hasattr(out, "decode"):
        out = out.decode()

    (sdist,) = tmpdir.visit("*.tar.gz")

    with tarfile.open(str(sdist), "r:gz") as tar:
        start = tar.getnames()[0] + "/"
        version = start[9:-1]
        simpler = {n.split("/", 1)[-1] for n in tar.getnames()[1:]}

        with contextlib.closing(
            tar.extractfile(tar.getmember(start + "setup.py"))
        ) as f:
            setup_py = f.read()

        with contextlib.closing(
            tar.extractfile(tar.getmember(start + "pyproject.toml"))
        ) as f:
            pyproject_toml = f.read()

        with contextlib.closing(
            tar.extractfile(
                tar.getmember(
                    start + "pybind11/share/cmake/pybind11/pybind11Config.cmake"
                )
            )
        ) as f:
            contents = f.read().decode("utf8")
        assert 'set(pybind11_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")' in contents

    files = {f"pybind11/{n}" for n in all_files}
    files |= sdist_files
    files |= {f"pybind11{n}" for n in local_sdist_files}
    files.add("pybind11.egg-info/entry_points.txt")
    files.add("pybind11.egg-info/requires.txt")
    assert simpler == files

    with open(os.path.join(MAIN_DIR, "tools", "setup_main.py.in"), "rb") as f:
        contents = (
            string.Template(f.read().decode())
            .substitute(version=version, extra_cmd="")
            .encode()
        )
    assert setup_py == contents

    with open(os.path.join(MAIN_DIR, "tools", "pyproject.toml"), "rb") as f:
        contents = f.read()
    assert pyproject_toml == contents


def test_build_global_dist(monkeypatch, tmpdir):

    monkeypatch.chdir(MAIN_DIR)
    monkeypatch.setenv("PYBIND11_GLOBAL_SDIST", "1")
    out = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--outdir",
            str(tmpdir),
        ]
    )

    if hasattr(out, "decode"):
        out = out.decode()

    (sdist,) = tmpdir.visit("*.tar.gz")

    with tarfile.open(str(sdist), "r:gz") as tar:
        start = tar.getnames()[0] + "/"
        version = start[16:-1]
        simpler = {n.split("/", 1)[-1] for n in tar.getnames()[1:]}

        with contextlib.closing(
            tar.extractfile(tar.getmember(start + "setup.py"))
        ) as f:
            setup_py = f.read()

        with contextlib.closing(
            tar.extractfile(tar.getmember(start + "pyproject.toml"))
        ) as f:
            pyproject_toml = f.read()

    files = {f"pybind11/{n}" for n in all_files}
    files |= sdist_files
    files |= {f"pybind11_global{n}" for n in local_sdist_files}
    assert simpler == files

    with open(os.path.join(MAIN_DIR, "tools", "setup_global.py.in"), "rb") as f:
        contents = (
            string.Template(f.read().decode())
            .substitute(version=version, extra_cmd="")
            .encode()
        )
        assert setup_py == contents

    with open(os.path.join(MAIN_DIR, "tools", "pyproject.toml"), "rb") as f:
        contents = f.read()
        assert pyproject_toml == contents


def tests_build_wheel(monkeypatch, tmpdir):
    monkeypatch.chdir(MAIN_DIR)

    subprocess.check_output(
        [sys.executable, "-m", "pip", "wheel", ".", "-w", str(tmpdir)]
    )

    (wheel,) = tmpdir.visit("*.whl")

    files = {f"pybind11/{n}" for n in all_files}
    files |= {
        "dist-info/LICENSE",
        "dist-info/METADATA",
        "dist-info/RECORD",
        "dist-info/WHEEL",
        "dist-info/entry_points.txt",
        "dist-info/top_level.txt",
    }

    with zipfile.ZipFile(str(wheel)) as z:
        names = z.namelist()

    trimmed = {n for n in names if "dist-info" not in n}
    trimmed |= {f"dist-info/{n.split('/', 1)[-1]}" for n in names if "dist-info" in n}
    assert files == trimmed


def tests_build_global_wheel(monkeypatch, tmpdir):
    monkeypatch.chdir(MAIN_DIR)
    monkeypatch.setenv("PYBIND11_GLOBAL_SDIST", "1")

    subprocess.check_output(
        [sys.executable, "-m", "pip", "wheel", ".", "-w", str(tmpdir)]
    )

    (wheel,) = tmpdir.visit("*.whl")

    files = {f"data/data/{n}" for n in src_files}
    files |= {f"data/headers/{n[8:]}" for n in headers}
    files |= {
        "dist-info/LICENSE",
        "dist-info/METADATA",
        "dist-info/WHEEL",
        "dist-info/top_level.txt",
        "dist-info/RECORD",
    }

    with zipfile.ZipFile(str(wheel)) as z:
        names = z.namelist()

    beginning = names[0].split("/", 1)[0].rsplit(".", 1)[0]
    trimmed = {n[len(beginning) + 1 :] for n in names}

    assert files == trimmed
