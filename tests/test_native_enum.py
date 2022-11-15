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

ALTITUDE_MEMBERS = (
    ("high", "h"),
    ("low", "l"),
)

EXPORT_VALUES_MEMBERS = (
    ("exv0", 0),
    ("exv1", 1),
)

MEMBER_DOC_MEMBERS = (
    ("mem0", 0),
    ("mem1", 1),
    ("mem2", 2),
)


@pytest.mark.parametrize(
    "enum_type", (m.smallenum, m.color, m.altitude, m.export_values, m.member_doc)
)
def test_enum_type(enum_type):
    assert isinstance(enum_type, enum.EnumMeta)


@pytest.mark.parametrize(
    "enum_type,members",
    (
        (m.smallenum, SMALLENUM_MEMBERS),
        (m.color, COLOR_MEMBERS),
        (m.altitude, ALTITUDE_MEMBERS),
        (m.export_values, EXPORT_VALUES_MEMBERS),
        (m.member_doc, MEMBER_DOC_MEMBERS),
    ),
)
def test_enum_members(enum_type, members):
    for name, value in members:
        assert enum_type[name].value == value


def test_export_values():
    assert m.exv0 is m.export_values.exv0
    assert m.exv1 is m.export_values.exv1


def test_member_doc():
    pure_native = enum.IntEnum("pure_native", (("mem", 0),))
    assert m.member_doc.mem0.__doc__ == "docA"
    assert m.member_doc.mem1.__doc__ == pure_native.mem.__doc__
    assert m.member_doc.mem2.__doc__ == "docC"


def test_pybind11_isinstance_color():
    for name, _ in COLOR_MEMBERS:
        assert m.isinstance_color(m.color[name])
    assert not m.isinstance_color(m.color)
    for name, _ in SMALLENUM_MEMBERS:
        assert not m.isinstance_color(m.smallenum[name])
    assert not m.isinstance_color(m.smallenum)
    assert not m.isinstance_color(None)


def test_pass_color_success():
    for name, value in COLOR_MEMBERS:
        assert m.pass_color(m.color[name]) == value


def test_pass_color_fail():
    with pytest.raises(TypeError) as excinfo:
        m.pass_color(None)
    assert "test_native_enum::color" in str(excinfo.value)


def test_return_color_success():
    for name, value in COLOR_MEMBERS:
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


@pytest.mark.skipif(isinstance(m.obj_cast_color_ptr, str), reason=m.obj_cast_color_ptr)
def test_obj_cast_color_ptr():
    with pytest.raises(RuntimeError) as excinfo:
        m.obj_cast_color_ptr(m.color.red)
    assert str(excinfo.value) == "Unable to cast native enum type to reference"
