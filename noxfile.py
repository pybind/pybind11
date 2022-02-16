import os
import sys
from pathlib import Path

import nox

nox.needs_version = ">=2022.1.7"
nox.options.sessions = ["lint", "tests", "tests_packaging"]

DIR = Path(__file__).parent.resolve()

PYTHON_VERISONS = [
    "3.6",
    "3.7",
    "3.8",
    "3.9",
    "3.10",
    "3.11",
    "pypy3.7",
    "pypy3.8",
    "pypy3.9",
]
PYPY_VERSIONS = ["3.7", "3.8", "3.9"]

if os.environ.get("CI", None):
    nox.options.error_on_missing_interpreters = True


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """
    Lint the codebase (except for clang-format/tidy).
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "-a")


@nox.session(python=PYTHON_VERISONS)
def tests(session: nox.Session) -> None:
    """
    Run the tests (requires a compiler).
    """
    tmpdir = session.create_tmp()
    session.install("cmake")
    session.install("-r", "tests/requirements.txt")
    session.run(
        "cmake",
        "-S.",
        f"-B{tmpdir}",
        "-DPYBIND11_WERROR=ON",
        "-DDOWNLOAD_CATCH=ON",
        "-DDOWNLOAD_EIGEN=ON",
        *session.posargs,
    )
    session.run("cmake", "--build", tmpdir)
    session.run("cmake", "--build", tmpdir, "--config=Release", "--target", "check")


@nox.session
def tests_packaging(session: nox.Session) -> None:
    """
    Run the packaging tests.
    """

    session.install("-r", "tests/requirements.txt", "--prefer-binary")
    session.run("pytest", "tests/extra_python_package")


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass "serve" to serve.
    """

    session.install("-r", "docs/requirements.txt")
    session.chdir("docs")

    if "pdf" in session.posargs:
        session.run("sphinx-build", "-b", "latexpdf", ".", "_build")
        return

    session.run("sphinx-build", "-b", "html", ".", "_build")

    if "serve" in session.posargs:
        session.log("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")
        session.run("python", "-m", "http.server", "8000", "-d", "_build/html")
    elif session.posargs:
        session.error("Unsupported argument to docs")


@nox.session(reuse_venv=True)
def make_changelog(session: nox.Session) -> None:
    """
    Inspect the closed issues and make entries for a changelog.
    """
    session.install("ghapi", "rich")
    session.run("python", "tools/make_changelog.py")


@nox.session(reuse_venv=True)
def build(session: nox.Session) -> None:
    """
    Build SDists and wheels.
    """

    session.install("build")
    session.log("Building normal files")
    session.run("python", "-m", "build", *session.posargs)
    session.log("Building pybind11-global files (PYBIND11_GLOBAL_SDIST=1)")
    session.run(
        "python", "-m", "build", *session.posargs, env={"PYBIND11_GLOBAL_SDIST": "1"}
    )


@nox.session
@nox.parametrize("pypy", PYPY_VERSIONS, ids=PYPY_VERSIONS)
def pypy_upstream(session: nox.Session, pypy: str) -> None:
    """
    Test against upstream PyPy (64-bit UNIX only)
    """
    import tarfile
    import urllib.request

    binary = "linux64" if sys.platform.startswith("linux") else "osx64"
    url = (
        f"https://buildbot.pypy.org/nightly/py{pypy}/pypy-c-jit-latest-{binary}.tar.bz2"
    )

    tmpdir = session.create_tmp()
    with session.chdir(tmpdir):
        urllib.request.urlretrieve(url, "pypy.tar.bz2")
        with tarfile.open("pypy.tar.bz2", "r:bz2") as tar:
            tar.extractall()
    (found,) = Path(tmpdir).glob("*/bin/pypy3")
    pypy_prog = str(found.resolve())
    pypy_dir = found.parent.parent

    session.run(pypy_prog, "-m", "ensurepip", external=True)
    session.run(pypy_prog, "-m", "pip", "install", "--upgrade", "pip", external=True)
    session.run(
        pypy_prog,
        "-m",
        "pip",
        "install",
        "pytest",
        "numpy;python_version<'3.9' and platform_system=='Linux'",
        "--only-binary=:all:",
        external=True,
    )

    session.install("cmake", "ninja")
    build_dir = session.create_tmp()
    tmpdir = session.create_tmp()
    session.run(
        "cmake",
        f"-S{DIR}",
        f"-B{build_dir}",
        "-DPYBIND11_FINDPYTHON=ON",
        f"-DPython_ROOT={pypy_dir}",
        "-GNinja",
        "-DPYBIND11_WERROR=ON",
        "-DDOWNLOAD_EIGEN=ON",
        *session.posargs,
    )
    session.run("cmake", "--build", build_dir)
    session.run("cmake", "--build", build_dir, "--target", "pytest")
    session.run("cmake", "--build", build_dir, "--target", "test_cmake_build")
