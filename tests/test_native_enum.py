from __future__ import annotations

import enum
import pickle

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

CLASS_WITH_ENUM_IN_CLASS_MEMBERS = (
    ("one", 0),
    ("two", 1),
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

ENUM_TYPES_AND_MEMBERS = (
    (m.smallenum, SMALLENUM_MEMBERS),
    (m.color, COLOR_MEMBERS),
    (m.altitude, ALTITUDE_MEMBERS),
    (m.export_values, EXPORT_VALUES_MEMBERS),
    (m.member_doc, MEMBER_DOC_MEMBERS),
    (m.class_with_enum.in_class, CLASS_WITH_ENUM_IN_CLASS_MEMBERS),
)

ENUM_TYPES = [_[0] for _ in ENUM_TYPES_AND_MEMBERS]


@pytest.mark.parametrize("enum_type", ENUM_TYPES)
def test_enum_type(enum_type):
    assert isinstance(enum_type, enum.EnumMeta)
    assert enum_type.__module__ == m.__name__


@pytest.mark.parametrize(("enum_type", "members"), ENUM_TYPES_AND_MEMBERS)
def test_enum_members(enum_type, members):
    for name, value in members:
        assert enum_type[name].value == value


@pytest.mark.parametrize(("enum_type", "members"), ENUM_TYPES_AND_MEMBERS)
def test_pickle_roundtrip(enum_type, members):
    for name, _ in members:
        orig = enum_type[name]
        if enum_type is m.class_with_enum.in_class:
            # This is a general pickle limitation.
            with pytest.raises(pickle.PicklingError):
                pickle.dumps(orig)
        else:
            # This only works if __module__ is correct.
            serialized = pickle.dumps(orig)
            restored = pickle.loads(serialized)
            assert restored == orig


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


def test_py_cast_color_handle():
    for name, value in COLOR_MEMBERS:
        assert m.py_cast_color_handle(m.color[name]) == value


def test_native_enum_data_was_not_added_error_message():
    msg = m.native_enum_data_was_not_added_error_message("Fake")
    assert msg == (
        "`native_enum` was not added to any module."
        ' Use e.g. `m += native_enum<...>("Fake", ...)` to fix.'
    )


@pytest.mark.parametrize(
    "func", [m.native_enum_ctor_malformed_utf8, m.native_enum_value_malformed_utf8]
)
def test_native_enum_malformed_utf8(func):
    malformed_utf8 = b"\x80"
    with pytest.raises(UnicodeDecodeError):
        func(malformed_utf8)


def test_double_registration_native_enum():
    with pytest.raises(RuntimeError) as excinfo:
        m.double_registration_native_enum(m)
    assert (
        str(excinfo.value)
        == 'pybind11::native_enum<...>("fake_double_registration_native_enum") is already registered!'
    )


def test_native_enum_name_clash():
    m.fake_native_enum_name_clash = None
    with pytest.raises(RuntimeError) as excinfo:
        m.native_enum_name_clash(m)
    assert (
        str(excinfo.value)
        == 'pybind11::native_enum<...>("fake_native_enum_name_clash"):'
        " an object with that name is already defined"
    )


def test_native_enum_value_name_clash():
    m.fake_native_enum_value_name_clash_x = None
    with pytest.raises(RuntimeError) as excinfo:
        m.native_enum_value_name_clash(m)
    assert (
        str(excinfo.value)
        == 'pybind11::native_enum<...>("fake_native_enum_value_name_clash")'
        '.value("fake_native_enum_value_name_clash_x"):'
        " an object with that name is already defined"
    )


def test_double_registration_enum_before_native_enum():
    with pytest.raises(RuntimeError) as excinfo:
        m.double_registration_enum_before_native_enum(m)
    assert (
        str(excinfo.value)
        == 'pybind11::native_enum<...>("fake_enum_first") is already registered'
        " as a `pybind11::enum_` or `pybind11::class_`!"
    )


def test_double_registration_native_enum_before_enum():
    with pytest.raises(RuntimeError) as excinfo:
        m.double_registration_native_enum_before_enum(m)
    assert (
        str(excinfo.value)
        == 'pybind11::enum_ "name_must_be_different_to_reach_desired_code_path"'
        " is already registered as a pybind11::native_enum!"
    )


def test_native_enum_correct_use_failure():
    if not isinstance(m.native_enum_correct_use_failure, str):
        m.native_enum_correct_use_failure()
        pytest.fail("Process termination expected.")
