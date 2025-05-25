from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import pybind11_stubgen
import pytest
from mypy import api

from pybind11_tests import stubgen as m


class MypyResult(NamedTuple):
    normal_report: str
    error_report: str
    exit_status: int


def run_mypy(stubs: Path) -> MypyResult:
    """Run mypy on the given stubs directory."""
    normal_report, error_report, exit_status = api.run(
        [stubs.as_posix(), "--no-color-output"]
    )
    print("Normal report:")
    print(normal_report)
    print("Error report:")
    print(error_report)
    return MypyResult(normal_report, error_report, exit_status)


@pytest.mark.xfail(
    sys.version_info >= (3, 14), reason="mypy does not support Python 3.14+ yet"
)
def test_stubgen(tmp_path: Path) -> None:
    assert m.add_int(1, 2) == 3
    # Generate stub into temporary directory
    pybind11_stubgen.main(
        [
            "pybind11_tests.stubgen",
            "-o",
            tmp_path.as_posix(),
        ]
    )
    # Check stub file is generated and contains expected content
    stub_file = tmp_path / "pybind11_tests" / "stubgen.pyi"
    assert stub_file.exists()
    stub_content = stub_file.read_text()
    assert (
        "def add_int(a: typing.SupportsInt, b: typing.SupportsInt) -> int:"
        in stub_content
    )
    # Run mypy on the generated stub file
    result = run_mypy(stub_file)
    assert result.exit_status == 0
    assert "Success: no issues found in 1 source file" in result.normal_report


def test_stubgen_all(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Generate stub into temporary directory
    pybind11_stubgen.main(
        [
            "pybind11_tests",
            "-o",
            tmp_path.as_posix(),
        ]
    )
    # Errors are reported using logging
    assert (
        "Raw C++ types/values were found in signatures extracted from docstrings"
        in caplog.text
    )
    # Check stub file is generated and contains expected content
    stubs = tmp_path / "pybind11_tests"
    assert stubs.exists()
    # Run mypy on the generated stub file
    result = run_mypy(stubs)
    assert result.exit_status == 0
    assert "Success: no issues found in 1 source file" in result.normal_report
