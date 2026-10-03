from __future__ import annotations

import os
import re
import subprocess
import sys
import venv
from pathlib import Path

DIR = Path(__file__).parent.resolve()
MAIN_DIR = DIR.parent.parent

# The newest release used as the cross-version ABI baseline, and the
# PYBIND11_INTERNALS_VERSION it ships with.
BASELINE_VERSION = "3.0.4"
BASELINE_INTERNALS_VERSION = 11

# The internals version of this checkout. If test_internals_version_pinned
# fails, the ABI was bumped: make sure that was intentional, then update this
# number (and the baseline above once a compatible release exists).
EXPECTED_INTERNALS_VERSION = 12


def read_internals_version() -> int:
    header = MAIN_DIR / "include/pybind11/detail/internals.h"
    match = re.search(
        r"^#\s*define\s+PYBIND11_INTERNALS_VERSION\s+(\d+)",
        header.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert match, "PYBIND11_INTERNALS_VERSION not found in internals.h"
    return int(match.group(1))


def test_internals_version_pinned():
    """Any bump of PYBIND11_INTERNALS_VERSION must be a conscious decision."""
    assert read_internals_version() == EXPECTED_INTERNALS_VERSION


def test_cross_version_abi(tmp_path: Path) -> None:
    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True)
    bin_dir = "Scripts" if sys.platform.startswith("win") else "bin"
    python = venv_dir / bin_dir / "python"

    def run(*args: str, name: str | None = None) -> None:
        env = os.environ.copy()
        if name is not None:
            env["EXAMPLE_NAME"] = name
        subprocess.run([os.fspath(python), *args], check=True, env=env)

    # Build pet against the baseline release; no build isolation, so that the
    # pinned pybind11 (not the latest release) provides the headers.
    run("-m", "pip", "install", f"pybind11=={BASELINE_VERSION}", "setuptools>=70.1")
    run("-m", "pip", "install", os.fspath(DIR), "--no-build-isolation", name="pet")

    # Build dog against this checkout.
    run("-m", "pip", "install", os.fspath(MAIN_DIR))
    run("-m", "pip", "install", os.fspath(DIR), "--no-build-isolation", name="dog")

    check_args = []
    if BASELINE_INTERNALS_VERSION != EXPECTED_INTERNALS_VERSION:
        check_args.append("--expect-incompatible")
    run(os.fspath(DIR / "check_installed.py"), *check_args)
