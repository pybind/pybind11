from __future__ import annotations

import gc

import pytest

from pybind11_tests import class_release_gil_before_calling_cpp_dtor as m


@pytest.mark.parametrize(
    ("probe_type", "unique_key", "expected_result"),
    [
        (m.ProbeType0, "without_manipulating_gil", "1"),
        (m.ProbeType1, "release_gil_before_calling_cpp_dtor", "0"),
    ],
)
def test_gil_state_check_results(probe_type, unique_key, expected_result):
    probe_type(unique_key)
    gc.collect()
    result = m.PopPyGILState_Check_Result(unique_key)
    assert result == expected_result
