from __future__ import annotations

from pathlib import Path

import pybind11_stubgen
from mypy import api

from pybind11_tests import stubgen as m


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
    normal_report, error_report, exit_status = api.run([stub_file.as_posix()])
    print("Normal report:")
    print(normal_report)
    print("Error report:")
    print(error_report)
    assert exit_status == 0
    assert "Success: no issues found in 1 source file" in normal_report
