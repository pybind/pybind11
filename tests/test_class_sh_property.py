# The compact 4-character naming scheme (e.g. mptr, cptr, shcp) is explained at the top of
# test_class_sh_property.cpp.

import pytest

import env  # noqa: F401
from pybind11_tests import class_sh_property as m


@pytest.mark.xfail("env.PYPY", reason="gc after `del field` is apparently deferred")
@pytest.mark.parametrize("m_attr", ["m_valu_readonly", "m_valu_readwrite"])
def test_valu_getter(m_attr):
    # Reduced from PyCLIF test:
    # https://github.com/google/clif/blob/c371a6d4b28d25d53a16e6d2a6d97305fb1be25a/clif/testing/python/nested_fields_test.py#L56
    outer = m.Outer()
    field = getattr(outer, m_attr)
    assert field.num == -99
    with pytest.raises(ValueError) as excinfo:
        m.DisownOuter(outer)
    assert str(excinfo.value) == "Cannot disown use_count != 1 (loaded_as_unique_ptr)."
    del field
    m.DisownOuter(outer)
    with pytest.raises(ValueError, match="Python instance was disowned") as excinfo:
        getattr(outer, m_attr)


def test_valu_setter():
    outer = m.Outer()
    assert outer.m_valu_readonly.num == -99
    assert outer.m_valu_readwrite.num == -99
    field = m.Field()
    field.num = 35
    outer.m_valu_readwrite = field
    assert outer.m_valu_readonly.num == 35
    assert outer.m_valu_readwrite.num == 35


@pytest.mark.parametrize("m_attr", ["m_shmp", "m_shcp"])
def test_shp(m_attr):
    m_attr_readonly = m_attr + "_readonly"
    m_attr_readwrite = m_attr + "_readwrite"
    outer = m.Outer()
    assert getattr(outer, m_attr_readonly) is None
    assert getattr(outer, m_attr_readwrite) is None
    field = m.Field()
    field.num = 43
    setattr(outer, m_attr_readwrite, field)
    assert getattr(outer, m_attr_readonly).num == 43
    assert getattr(outer, m_attr_readwrite).num == 43
    getattr(outer, m_attr_readonly).num = 57
    getattr(outer, m_attr_readwrite).num = 57
    assert field.num == 57
    del field
    assert getattr(outer, m_attr_readonly).num == 57
    assert getattr(outer, m_attr_readwrite).num == 57


@pytest.mark.parametrize(
    ("field_type", "num_default", "outer_type"),
    [
        (m.ClassicField, -88, m.ClassicOuter),
        (m.Field, -99, m.Outer),
    ],
)
@pytest.mark.parametrize("m_attr", ["m_mptr", "m_cptr"])
@pytest.mark.parametrize("r_kind", ["_readonly", "_readwrite"])
def test_ptr(field_type, num_default, outer_type, m_attr, r_kind):
    m_attr_r_kind = m_attr + r_kind
    outer = outer_type()
    assert getattr(outer, m_attr_r_kind) is None
    field = field_type()
    assert field.num == num_default
    setattr(outer, m_attr + "_readwrite", field)
    assert getattr(outer, m_attr_r_kind).num == num_default
    field.num = 76
    assert getattr(outer, m_attr_r_kind).num == 76
    # Change to -88 or -99 to demonstrate Undefined Behavior (dangling pointer).
    if num_default == 88 and m_attr == "m_mptr":
        del field
    assert getattr(outer, m_attr_r_kind).num == 76


@pytest.mark.parametrize("m_attr_readwrite", ["m_uqmp_readwrite", "m_uqcp_readwrite"])
def test_uqp(m_attr_readwrite):
    outer = m.Outer()
    assert getattr(outer, m_attr_readwrite) is None
    field_orig = m.Field()
    field_orig.num = 39
    setattr(outer, m_attr_readwrite, field_orig)
    with pytest.raises(ValueError, match="Python instance was disowned"):
        _ = field_orig.num
    field_retr1 = getattr(outer, m_attr_readwrite)
    assert getattr(outer, m_attr_readwrite) is None
    assert field_retr1.num == 39
    field_retr1.num = 93
    setattr(outer, m_attr_readwrite, field_retr1)
    with pytest.raises(ValueError):
        _ = field_retr1.num
    field_retr2 = getattr(outer, m_attr_readwrite)
    assert field_retr2.num == 93


# Proof-of-concept (POC) for safe & intuitive Python access to unique_ptr members.
# The C++ member unique_ptr is disowned to a temporary Python object for accessing
# an attribute of the member. After the attribute was accessed, the Python object
# is disowned back to the C++ member unique_ptr.
# Productizing this POC is left for a future separate PR, as needed.
class unique_ptr_field_proxy_poc:
    def __init__(self, obj, field_name):
        object.__setattr__(self, "__obj", obj)
        object.__setattr__(self, "__field_name", field_name)

    def __getattr__(self, *args, **kwargs):
        return _proxy_dereference(self, getattr, *args, **kwargs)

    def __setattr__(self, *args, **kwargs):
        return _proxy_dereference(self, setattr, *args, **kwargs)

    def __delattr__(self, *args, **kwargs):
        return _proxy_dereference(self, delattr, *args, **kwargs)


def _proxy_dereference(proxy, xxxattr, *args, **kwargs):
    obj = object.__getattribute__(proxy, "__obj")
    field_name = object.__getattribute__(proxy, "__field_name")
    field = getattr(obj, field_name)  # Disowns the C++ unique_ptr member.
    assert field is not None
    try:
        return xxxattr(field, *args, **kwargs)
    finally:
        setattr(obj, field_name, field)  # Disowns the temporary Python object (field).


@pytest.mark.parametrize("m_attr", ["m_uqmp", "m_uqcp"])
def test_unique_ptr_field_proxy_poc(m_attr):
    m_attr_readwrite = m_attr + "_readwrite"
    outer = m.Outer()
    field_orig = m.Field()
    field_orig.num = 45
    setattr(outer, m_attr_readwrite, field_orig)
    field_proxy = unique_ptr_field_proxy_poc(outer, m_attr_readwrite)
    assert field_proxy.num == 45
    assert field_proxy.num == 45
    with pytest.raises(AttributeError):
        _ = field_proxy.xyz
    assert field_proxy.num == 45
    field_proxy.num = 82
    assert field_proxy.num == 82
    field_proxy = unique_ptr_field_proxy_poc(outer, m_attr_readwrite)
    assert field_proxy.num == 82
    with pytest.raises(AttributeError):
        del field_proxy.num
    assert field_proxy.num == 82
