import pytest

import pybind11_tests.odr_guard_2 as m


def test_sizeof_mrc_odr_guard():
    if hasattr(m, "sizeof_mrc_odr_guard"):
        assert m.sizeof_mrc_odr_guard() == 8
    else:
        pytest.skip("sizeof_mrc_odr_guard")


def test_type_mrc_to_python():
    if hasattr(m, "type_mrc_to_python"):
        assert m.type_mrc_to_python() == 2222
    else:
        pytest.skip("type_mrc_to_python")


def test_type_mrc_from_python():
    if hasattr(m, "type_mrc_from_python"):
        assert m.type_mrc_from_python("ignored") == 222
    else:
        pytest.skip("type_mrc_from_python")


def test_mrc_odr_guard():
    if hasattr(m, "mrc_odr_guard"):
        i = m.mrc_odr_guard()
        m.type_mrc_to_python()
        j = m.mrc_odr_guard()
        assert j == i + 1
    else:
        pytest.skip("mrc_odr_guard")
