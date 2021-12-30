# -*- coding: utf-8 -*-
import pytest

import env  # noqa: F401
from pybind11_tests import class_sh_property as m


@pytest.mark.xfail("env.PYPY", reason="gc after `del inner` is apparently deferred")
def test_field_getter(msg):
    # Reduced from PyCLIF test:
    # https://github.com/google/clif/blob/c371a6d4b28d25d53a16e6d2a6d97305fb1be25a/clif/testing/python/nested_fields_test.py#L56
    outer = m.Outer()
    inner = outer.field
    assert inner.value == -99
    with pytest.raises(ValueError) as excinfo:
        m.DisownOuter(outer)
    assert msg(excinfo.value) == "Cannot disown use_count != 1 (loaded_as_unique_ptr)."
    del inner
    m.DisownOuter(outer)
    with pytest.raises(ValueError) as excinfo:
        outer.field
    assert (
        msg(excinfo.value)
        == "Missing value for wrapped C++ type: Python instance was disowned."
    )


def test_field_setter(msg):
    outer = m.Outer()
    assert outer.field.value == -99
    field = m.Inner()
    field.value = 35
    outer.field = field
    assert outer.field.value == 35
