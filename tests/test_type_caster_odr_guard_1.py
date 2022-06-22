import pytest

import pybind11_tests.odr_guard_1 as m


def test_type_mrc_to_python():
    assert m.type_mrc_to_python() == 1111


def test_type_mrc_from_python():
    assert m.type_mrc_from_python("ignored") == 111


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
    else:
        assert num_violations == 1
