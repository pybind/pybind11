import enum

import pytest

from pybind11_tests import native_enum as m

COLOR_MEMBERS = (
    ("red", 0),
    ("yellow", 1),
    ("green", 20),
    ("blue", 21),
)


def test_enum_color_type():
    assert isinstance(m.color, enum.EnumMeta)


@pytest.mark.parametrize("name,value", COLOR_MEMBERS)
def test_enum_color_members(name, value):
    assert m.color[name] == value


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
