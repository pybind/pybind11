import pytest

from pybind11_tests import type_fingerprint as m


def test_std_string():
    pytest.skip(f"SHOW: {m.std_string()}")
