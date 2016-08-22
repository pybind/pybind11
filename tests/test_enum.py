import pytest


def test_unscoped_enum():
    from pybind11_tests import UnscopedEnum, EOne

    assert str(UnscopedEnum.EOne) == "UnscopedEnum.EOne"
    assert str(UnscopedEnum.ETwo) == "UnscopedEnum.ETwo"
    assert str(EOne) == "UnscopedEnum.EOne"

    # no TypeError exception for unscoped enum ==/!= int comparisons
    y = UnscopedEnum.ETwo
    assert y == 2
    assert y != 3

    assert int(UnscopedEnum.ETwo) == 2
    assert str(UnscopedEnum(2)) == "UnscopedEnum.ETwo"


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
