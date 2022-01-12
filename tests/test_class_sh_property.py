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


@pytest.mark.parametrize("m_attr_disown", ("m_uqmp_disown", "m_uqcp_disown"))
def test_uqp(m_attr_disown, msg):
    outer = m.Outer()
    assert getattr(outer, m_attr_disown) is None
    field_orig = m.Field()
    field_orig.num = 39
    setattr(outer, m_attr_disown, field_orig)
    with pytest.raises(ValueError) as excinfo:
        field_orig.num
    assert (
        msg(excinfo.value)
        == "Missing value for wrapped C++ type: Python instance was disowned."
    )
    field_retr1 = getattr(outer, m_attr_disown)
    assert getattr(outer, m_attr_disown) is None
    assert field_retr1.num == 39
    field_retr1.num = 93
    setattr(outer, m_attr_disown, field_retr1)
    with pytest.raises(ValueError):
        field_retr1.num
    field_retr2 = getattr(outer, m_attr_disown)
    assert field_retr2.num == 93


def _dereference(proxy, xxxattr, *args, **kwargs):
    obj = object.__getattribute__(proxy, "__obj")
    field_name = object.__getattribute__(proxy, "__field_name")
    field = getattr(obj, field_name)
    assert field is not None
    try:
        return xxxattr(field, *args, **kwargs)
    finally:
        setattr(obj, field_name, field)


class unique_ptr_field_proxy_poc(object):  # noqa: N801
    def __init__(self, obj, field_name):
        object.__setattr__(self, "__obj", obj)
        object.__setattr__(self, "__field_name", field_name)

    def __getattr__(self, *args, **kwargs):
        return _dereference(self, getattr, *args, **kwargs)

    def __setattr__(self, *args, **kwargs):
        return _dereference(self, setattr, *args, **kwargs)

    def __delattr__(self, *args, **kwargs):
        return _dereference(self, delattr, *args, **kwargs)


@pytest.mark.parametrize("m_attr_disown", ("m_uqmp_disown", "m_uqcp_disown"))
def test_unique_ptr_field_proxy_poc(m_attr_disown, msg):
    outer = m.Outer()
    field_orig = m.Field()
    field_orig.num = 45
    setattr(outer, m_attr_disown, field_orig)
    field_proxy = unique_ptr_field_proxy_poc(outer, m_attr_disown)
    assert field_proxy.num == 45
    assert field_proxy.num == 45
    with pytest.raises(AttributeError):
        field_proxy.xyz
    assert field_proxy.num == 45
    field_proxy.num = 82
    assert field_proxy.num == 82
    field_proxy = unique_ptr_field_proxy_poc(outer, m_attr_disown)
    assert field_proxy.num == 82
    with pytest.raises(AttributeError):
        del field_proxy.num
    assert field_proxy.num == 82


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
