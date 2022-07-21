import pytest

import pybind11_tests.type_caster_odr_guard_2 as m


def test_type_mrc_to_python():
    val = m.type_mrc_to_python()
    if val == 202 + 2020:
        pytest.skip(
            "UNEXPECTED: test_type_caster_odr_guard_2.cpp prevailed (to_python)."
        )
    else:
        assert val == 202 + 1010


def test_type_mrc_from_python():
    val = m.type_mrc_from_python("ignored")
    if val == 200 + 22:
        pytest.skip(
            "UNEXPECTED: test_type_caster_odr_guard_2.cpp prevailed (from_python)."
        )
    else:
        assert val == 200 + 11
