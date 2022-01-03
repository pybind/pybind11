# -*- coding: utf-8 -*-
import pytest

import env  # noqa: F401
from pybind11_tests import class_sh_property as m


@pytest.mark.xfail("env.PYPY", reason="gc after `del field` is apparently deferred")
def test_field_getter(msg):
    # Reduced from PyCLIF test:
    # https://github.com/google/clif/blob/c371a6d4b28d25d53a16e6d2a6d97305fb1be25a/clif/testing/python/nested_fields_test.py#L56
    outer = m.Outer()
    field = outer.m_val
    assert field.num == -99
    with pytest.raises(ValueError) as excinfo:
        m.DisownOuter(outer)
    assert msg(excinfo.value) == "Cannot disown use_count != 1 (loaded_as_unique_ptr)."
    del field
    m.DisownOuter(outer)
    with pytest.raises(ValueError) as excinfo:
        outer.m_val
    assert (
        msg(excinfo.value)
        == "Missing value for wrapped C++ type: Python instance was disowned."
    )


def test_field_setter(msg):
    outer = m.Outer()
    assert outer.m_val.num == -99
    field = m.Field()
    field.num = 35
    outer.m_val = field
    assert outer.m_val.num == 35
