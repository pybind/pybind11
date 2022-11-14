import enum

import pytest

from pybind11_tests import native_enum as m

SMALLENUM_MEMBERS = (
    ("a", 0),
    ("b", 1),
    ("c", 2),
)

COLOR_MEMBERS = (
    ("red", 0),
    ("yellow", 1),
    ("green", 20),
    ("blue", 21),
)


@pytest.mark.parametrize("enum_type", (m.smallenum, m.color))
def test_enum_color_type(enum_type):
    assert isinstance(enum_type, enum.EnumMeta)


def test_pybind11_isinstance_color():
    assert not m.isinstance_color(None)
    assert not m.isinstance_color(m.color)  # TODO: NEEDS FIXING


@pytest.mark.parametrize(
    "enum_type,members", ((m.smallenum, SMALLENUM_MEMBERS), (m.color, COLOR_MEMBERS))
)
def test_enum_color_members(enum_type, members):
    for name, value in members:
        assert enum_type[name] == value


@pytest.mark.parametrize("name,value", COLOR_MEMBERS)
def test_pass_color_success(name, value):
    assert m.pass_color(m.color[name]) == value


def test_pass_color_fail():
    with pytest.raises(TypeError) as excinfo:
        m.pass_color(None)
    assert "test_native_enum::color" in str(excinfo.value)


@pytest.mark.parametrize("name,value", COLOR_MEMBERS)
def test_return_color_success(name, value):
    assert m.return_color(value) == m.color[name]


def test_return_color_fail():
    with pytest.raises(ValueError) as excinfo_direct:
        m.color(2)
    with pytest.raises(ValueError) as excinfo_cast:
        m.return_color(2)
    assert str(excinfo_cast.value) == str(excinfo_direct.value)


def test_type_caster_enum_type_enabled_false():
    # This is really only a "does it compile" test.
    assert m.pass_some_proto_enum(None) is None
    assert m.return_some_proto_enum() is None


def test_obj_cast_color():
    assert m.obj_cast_color(m.color.green) == 1
    assert m.obj_cast_color(m.color.blue) == 0
    with pytest.raises(RuntimeError) as excinfo:
        m.obj_cast_color(None)
    assert str(excinfo.value).startswith("Unable to cast Python instance ")
