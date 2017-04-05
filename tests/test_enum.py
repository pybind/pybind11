import pytest


def test_unscoped_enum():
    from pybind11_tests import UnscopedEnum, EOne

    assert str(UnscopedEnum.EOne) == "UnscopedEnum.EOne"
    assert str(UnscopedEnum.ETwo) == "UnscopedEnum.ETwo"
    assert str(EOne) == "UnscopedEnum.EOne"
    # __members__ property
    assert UnscopedEnum.__members__ == {"EOne": UnscopedEnum.EOne, "ETwo": UnscopedEnum.ETwo}
    # __members__ readonly
    with pytest.raises(AttributeError):
        UnscopedEnum.__members__ = {}
    # __members__ returns a copy
    foo = UnscopedEnum.__members__
    foo["bar"] = "baz"
    assert UnscopedEnum.__members__ == {"EOne": UnscopedEnum.EOne, "ETwo": UnscopedEnum.ETwo}

    # no TypeError exception for unscoped enum ==/!= int comparisons
    y = UnscopedEnum.ETwo
    assert y == 2
    assert y != 3

    assert int(UnscopedEnum.ETwo) == 2
    assert str(UnscopedEnum(2)) == "UnscopedEnum.ETwo"

    # order
    assert UnscopedEnum.EOne < UnscopedEnum.ETwo
    assert UnscopedEnum.EOne < 2
    assert UnscopedEnum.ETwo > UnscopedEnum.EOne
    assert UnscopedEnum.ETwo > 1
    assert UnscopedEnum.ETwo <= 2
    assert UnscopedEnum.ETwo >= 2
    assert UnscopedEnum.EOne <= UnscopedEnum.ETwo
    assert UnscopedEnum.EOne <= 2
    assert UnscopedEnum.ETwo >= UnscopedEnum.EOne
    assert UnscopedEnum.ETwo >= 1
    assert not (UnscopedEnum.ETwo < UnscopedEnum.EOne)
    assert not (2 < UnscopedEnum.EOne)


def test_scoped_enum():
    from pybind11_tests import ScopedEnum, test_scoped_enum

    assert test_scoped_enum(ScopedEnum.Three) == "ScopedEnum::Three"
    z = ScopedEnum.Two
    assert test_scoped_enum(z) == "ScopedEnum::Two"

    # expected TypeError exceptions for scoped enum ==/!= int comparisons
    with pytest.raises(TypeError):
        assert z == 2
    with pytest.raises(TypeError):
        assert z != 3

    # order
    assert ScopedEnum.Two < ScopedEnum.Three
    assert ScopedEnum.Three > ScopedEnum.Two
    assert ScopedEnum.Two <= ScopedEnum.Three
    assert ScopedEnum.Two <= ScopedEnum.Two
    assert ScopedEnum.Two >= ScopedEnum.Two
    assert ScopedEnum.Three >= ScopedEnum.Two


def test_implicit_conversion():
    from pybind11_tests import ClassWithUnscopedEnum

    assert str(ClassWithUnscopedEnum.EMode.EFirstMode) == "EMode.EFirstMode"
    assert str(ClassWithUnscopedEnum.EFirstMode) == "EMode.EFirstMode"

    f = ClassWithUnscopedEnum.test_function
    first = ClassWithUnscopedEnum.EFirstMode
    second = ClassWithUnscopedEnum.ESecondMode

    assert f(first) == 1

    assert f(first) == f(first)
    assert not f(first) != f(first)

    assert f(first) != f(second)
    assert not f(first) == f(second)

    assert f(first) == int(f(first))
    assert not f(first) != int(f(first))

    assert f(first) != int(f(second))
    assert not f(first) == int(f(second))

    # noinspection PyDictCreation
    x = {f(first): 1, f(second): 2}
    x[f(first)] = 3
    x[f(second)] = 4
    # Hashing test
    assert str(x) == "{EMode.EFirstMode: 3, EMode.ESecondMode: 4}"


def test_binary_operators():
    from pybind11_tests import Flags

    assert int(Flags.Read) == 4
    assert int(Flags.Write) == 2
    assert int(Flags.Execute) == 1
    assert int(Flags.Read | Flags.Write | Flags.Execute) == 7
    assert int(Flags.Read | Flags.Write) == 6
    assert int(Flags.Read | Flags.Execute) == 5
    assert int(Flags.Write | Flags.Execute) == 3
    assert int(Flags.Write | 1) == 3

    state = Flags.Read | Flags.Write
    assert (state & Flags.Read) != 0
    assert (state & Flags.Write) != 0
    assert (state & Flags.Execute) == 0
    assert (state & 1) == 0

    state2 = ~state
    assert state2 == -7
    assert int(state ^ state2) == -1


def test_enum_to_int():
    from pybind11_tests import Flags, ClassWithUnscopedEnum
    from pybind11_tests import test_enum_to_int, test_enum_to_uint, test_enum_to_long_long

    test_enum_to_int(Flags.Read)
    test_enum_to_int(ClassWithUnscopedEnum.EMode.EFirstMode)
    test_enum_to_uint(Flags.Read)
    test_enum_to_uint(ClassWithUnscopedEnum.EMode.EFirstMode)
    test_enum_to_long_long(Flags.Read)
    test_enum_to_long_long(ClassWithUnscopedEnum.EMode.EFirstMode)


@pytest.requires_py3
def test_py3_enum():
    from pybind11_tests import (
        Py3Enum, Py3EnumEmpty, Py3EnumScoped,
        make_py3_enum, take_py3_enum, non_unique_py3_enum
    )

    from enum import IntEnum

    expected = {
        Py3Enum: [('A', -42), ('B', 1), ('C', 42)],
        Py3EnumEmpty: [],
        Py3EnumScoped: [('X', 10), ('Y', -1024)]
    }

    for tp, entries in expected.items():
        assert issubclass(tp, IntEnum)
        assert sorted(tp.__members__.items()) == entries

    assert make_py3_enum(True) is Py3EnumScoped.X
    assert make_py3_enum(False) is Py3EnumScoped.Y

    assert take_py3_enum(Py3EnumScoped.X)
    assert not take_py3_enum(Py3EnumScoped.Y)

    with pytest.raises(ValueError) as excinfo:
        non_unique_py3_enum()
    assert 'duplicate values found' in str(excinfo.value)
