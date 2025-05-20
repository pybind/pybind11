from __future__ import annotations

import sys
from pathlib import Path

import pybind11_stubgen
import pytest
from mypy import api

from pybind11_tests import stubgen_error as m


@pytest.mark.skipif(
    sys.version_info >= (3, 13), reason="CapsuleType available in 3.13+"
)
def test_stubgen(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Show stubgen/mypy errors for CapsuleType (not available in Python < 3.13)."""
    assert m.identity_capsule(None) is None
    # Generate stub into temporary directory
    pybind11_stubgen.main(
        [
            "pybind11_tests.stubgen_error",
            "-o",
            tmp_path.as_posix(),
        ]
    )
    # Errors are reported using logging
    assert "Can't find/import 'types.CapsuleType'" in caplog.text
    # Stub file is still generated if error is not fatal
    stub_file = tmp_path / "pybind11_tests" / "stubgen_error.pyi"
    assert stub_file.exists()
    stub_content = stub_file.read_text()
    assert (
        "def identity_capsule(c: types.CapsuleType) -> types.CapsuleType:"
        in stub_content
    )
    # Run mypy on the generated stub file
    # normal_report -> stdout, error_report -> stderr
    # Type errors seem to go into normal_report
    normal_report, error_report, exit_status = api.run([stub_file.as_posix()])
    print("Normal report:")
    print(normal_report)
    print("Error report:")
    print(error_report)
    assert exit_status == 1
    assert 'error: Name "types" is not defined  [name-defined]' in normal_report
