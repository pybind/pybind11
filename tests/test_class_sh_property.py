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


@pytest.mark.parametrize(
    "field_type, num_default, outer_type",
    [
        (m.ClassicField, -88, m.ClassicOuter),
        (m.Field, -99, m.Outer),
    ],
)
@pytest.mark.parametrize("m_attr", ("m_mptr", "m_cptr"))
def test_ptr(field_type, num_default, outer_type, m_attr):
    outer = outer_type()
    assert getattr(outer, m_attr) is None
    field = field_type()
    assert field.num == num_default
    setattr(outer, m_attr, field)
    assert getattr(outer, m_attr).num == num_default
    field.num = 76
    assert getattr(outer, m_attr).num == 76
    # Change to -88 or -99 to demonstrate Undefined Behavior (dangling pointer).
    if num_default == 88 and m_attr == "m_mptr":
        del field
    assert getattr(outer, m_attr).num == 76


@pytest.mark.xfail(
    "env.PYPY", reason="gc after `del field_co_own` is apparently deferred"
)
@pytest.mark.parametrize("m_attr", ("m_uqmp", "m_uqcp"))
def test_uqp(m_attr, msg):
    m_attr_disown = m_attr + "_disown"
    outer = m.Outer()
    assert getattr(outer, m_attr) is None
    assert getattr(outer, m_attr_disown) is None
    field = m.Field()
    field.num = 39
    setattr(outer, m_attr_disown, field)
    with pytest.raises(ValueError) as excinfo:
        field.num
    assert (
        msg(excinfo.value)
        == "Missing value for wrapped C++ type: Python instance was disowned."
    )
    field_co_own = getattr(outer, m_attr)
    assert getattr(outer, m_attr).num == 39
    assert field_co_own.num == 39
    # TODO: needs work.
    # with pytest.raises(RuntimeError) as excinfo:
    #     getattr(outer, m_attr_disown)
    # assert (
    #     msg(excinfo.value)
    #     == "Invalid unique_ptr: another instance owns this pointer already."
    # )
    del field_co_own
    field_excl_own = getattr(outer, m_attr_disown)
    assert getattr(outer, m_attr) is None
    assert field_excl_own.num == 39


@pytest.mark.parametrize("m_attr", ("m_shmp", "m_shcp"))
def test_shp(m_attr):
    outer = m.Outer()
    assert getattr(outer, m_attr) is None
    field = m.Field()
    field.num = 43
    setattr(outer, m_attr, field)
    assert getattr(outer, m_attr).num == 43
    getattr(outer, m_attr).num = 57
    assert field.num == 57
    del field
    assert getattr(outer, m_attr).num == 57
