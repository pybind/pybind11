# -*- coding: utf-8 -*-
import pytest

import env  # noqa: F401
from pybind11_tests import class_sh_property as m


@pytest.mark.xfail("env.PYPY", reason="gc after `del field` is apparently deferred")
def test_valu_getter(msg):
    # Reduced from PyCLIF test:
    # https://github.com/google/clif/blob/c371a6d4b28d25d53a16e6d2a6d97305fb1be25a/clif/testing/python/nested_fields_test.py#L56
    outer = m.Outer()
    field = outer.m_valu
    assert field.num == -99
    with pytest.raises(ValueError) as excinfo:
        m.DisownOuter(outer)
    assert msg(excinfo.value) == "Cannot disown use_count != 1 (loaded_as_unique_ptr)."
    del field
    m.DisownOuter(outer)
    with pytest.raises(ValueError) as excinfo:
        outer.m_valu
    assert (
        msg(excinfo.value)
        == "Missing value for wrapped C++ type: Python instance was disowned."
    )


def test_valu_setter():
    outer = m.Outer()
    assert outer.m_valu.num == -99
    field = m.Field()
    field.num = 35
    outer.m_valu = field
    assert outer.m_valu.num == 35


def test_uqmp(msg):
    outer = m.Outer()
    assert outer.m_uqmp is None
    field = m.Field()
    field.num = 39
    outer.m_uqmp_disown = field
    with pytest.raises(ValueError) as excinfo:
        field.num
    assert (
        msg(excinfo.value)
        == "Missing value for wrapped C++ type: Python instance was disowned."
    )
    field_getter = outer.m_uqmp
    assert outer.m_uqmp.num == 39
    assert field_getter.num == 39
