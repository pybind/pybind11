#!/usr/bin/env -S uv run

# /// script
# dependencies = ["nox>=2025.2.9"]
# ///

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

import nox

nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """
    Lint the codebase (except for clang-format/tidy).
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "-a", *session.posargs)


@nox.session
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

    session.install("-r", "tests/requirements.txt", "pip")
    session.run("pytest", "tests/extra_python_package", *session.posargs)


@nox.session(reuse_venv=True, default=False)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass --non-interactive to avoid serving.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    args, posargs = parser.parse_known_args(session.posargs)
    serve = args.builder == "html" and session.interactive

    extra_installs = ["sphinx-autobuild"] if serve else []
    session.install("-r", "docs/requirements.txt", *extra_installs)
    session.chdir("docs")

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        ".",
        f"_build/{args.builder}",
        *posargs,
    )

    if serve:
        session.run(
            "sphinx-autobuild", "--open-browser", "--ignore=.build", *shared_args
        )
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session(reuse_venv=True, default=False)
def make_changelog(session: nox.Session) -> None:
    """
    Inspect the closed issues and make entries for a changelog.
    """
    session.install_and_run_script("tools/make_changelog.py")


@nox.session(reuse_venv=True, default=False)
def build(session: nox.Session) -> None:
    """
    Build SDist and wheel.
    """

    session.install("build")
    session.log("Building normal files")
    session.run("python", "-m", "build", *session.posargs)


@contextlib.contextmanager
def preserve_file(filename: Path) -> Generator[str, None, None]:
    """
    Causes a file to be stored and preserved when the context manager exits.
    """
    old_stat = filename.stat()
    old_file = filename.read_text(encoding="utf-8")
    try:
        yield old_file
    finally:
        filename.write_text(old_file, encoding="utf-8")
        os.utime(filename, (old_stat.st_atime, old_stat.st_mtime))


@nox.session(reuse_venv=True)
def build_global(session: nox.Session) -> None:
    """
    Build global SDist and wheel.
    """

    installer = ["--installer=uv"] if session.venv_backend == "uv" else []
    session.install("build", "tomlkit")
    session.log("Building pybind11-global files")
    pyproject = Path("pyproject.toml")
    with preserve_file(pyproject):
        newer_txt = session.run("python", "tools/make_global.py", silent=True)
        assert isinstance(newer_txt, str)
        pyproject.write_text(newer_txt, encoding="utf-8")
        session.run(
            "python",
            "-m",
            "build",
            *installer,
            *session.posargs,
        )
