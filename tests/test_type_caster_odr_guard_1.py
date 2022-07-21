import pytest

import pybind11_tests
import pybind11_tests.type_caster_odr_guard_1 as m


def test_type_mrc_to_python():
    val = m.type_mrc_to_python()
    if val == 101 + 2020:
        pytest.skip(
            "UNEXPECTED: test_type_caster_odr_guard_2.cpp prevailed (to_python)."
        )
    else:
        assert val == 101 + 1010


def test_type_mrc_from_python():
    val = m.type_mrc_from_python("ignored")
    if val == 100 + 22:
        pytest.skip(
            "UNEXPECTED: test_type_caster_odr_guard_2.cpp prevailed (from_python)."
        )
    else:
        assert val == 100 + 11


def test_type_caster_odr_registry_values():
    reg_values = m.type_caster_odr_guard_registry_values()
    if reg_values is None:
        pytest.skip("type_caster_odr_guard_registry_values() is None")
    else:
        assert "test_type_caster_odr_guard_" in "\n".join(reg_values)


def test_type_caster_odr_violation_detected_counter():
    num_violations = m.type_caster_odr_violation_detected_count()
    if num_violations is None:
        pytest.skip("type_caster_odr_violation_detected_count() is None")
    elif num_violations == 0 and m.if_defined__NO_INLINE__:
        pytest.skip(
            "type_caster_odr_violation_detected_count() == 0: %s, %s, __NO_INLINE__"
            % (pybind11_tests.compiler_info, pybind11_tests.cpp_std)
        )
    else:
        assert num_violations == 1
