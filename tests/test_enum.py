import pytest
from pybind11_tests import enums as m


def test_unscoped_enum():
    assert str(m.UnscopedEnum.EOne) == "UnscopedEnum.EOne"
    assert str(m.UnscopedEnum.ETwo) == "UnscopedEnum.ETwo"
    assert str(m.EOne) == "UnscopedEnum.EOne"
    # __members__ property
    assert m.UnscopedEnum.__members__ == \
        {"EOne": m.UnscopedEnum.EOne, "ETwo": m.UnscopedEnum.ETwo}
    # __members__ readonly
    with pytest.raises(AttributeError):
        m.UnscopedEnum.__members__ = {}
    # __members__ returns a copy
    foo = m.UnscopedEnum.__members__
    foo["bar"] = "baz"
    assert m.UnscopedEnum.__members__ == \
        {"EOne": m.UnscopedEnum.EOne, "ETwo": m.UnscopedEnum.ETwo}

    assert m.UnscopedEnum.__doc__ == \
        '''An unscoped enumeration

Members:

  EOne : Docstring for EOne

  ETwo : Docstring for ETwo''' or m.UnscopedEnum.__doc__ == \
        '''An unscoped enumeration

Members:

  ETwo : Docstring for ETwo

  EOne : Docstring for EOne'''

    # no TypeError exception for unscoped enum ==/!= int comparisons
    y = m.UnscopedEnum.ETwo
    assert y == 2
    assert y != 3

    assert int(m.UnscopedEnum.ETwo) == 2
    assert str(m.UnscopedEnum(2)) == "UnscopedEnum.ETwo"

    # order
    assert m.UnscopedEnum.EOne < m.UnscopedEnum.ETwo
    assert m.UnscopedEnum.EOne < 2
    assert m.UnscopedEnum.ETwo > m.UnscopedEnum.EOne
    assert m.UnscopedEnum.ETwo > 1
    assert m.UnscopedEnum.ETwo <= 2
    assert m.UnscopedEnum.ETwo >= 2
    assert m.UnscopedEnum.EOne <= m.UnscopedEnum.ETwo
    assert m.UnscopedEnum.EOne <= 2
    assert m.UnscopedEnum.ETwo >= m.UnscopedEnum.EOne
    assert m.UnscopedEnum.ETwo >= 1
    assert not (m.UnscopedEnum.ETwo < m.UnscopedEnum.EOne)
    assert not (2 < m.UnscopedEnum.EOne)


def test_scoped_enum():
    assert m.test_scoped_enum(m.ScopedEnum.Three) == "ScopedEnum::Three"
    z = m.ScopedEnum.Two
    assert m.test_scoped_enum(z) == "ScopedEnum::Two"

    # expected TypeError exceptions for scoped enum ==/!= int comparisons
    with pytest.raises(TypeError):
        assert z == 2
    with pytest.raises(TypeError):
        assert z != 3

    # order
    assert m.ScopedEnum.Two < m.ScopedEnum.Three
    assert m.ScopedEnum.Three > m.ScopedEnum.Two
    assert m.ScopedEnum.Two <= m.ScopedEnum.Three
    assert m.ScopedEnum.Two <= m.ScopedEnum.Two
    assert m.ScopedEnum.Two >= m.ScopedEnum.Two
    assert m.ScopedEnum.Three >= m.ScopedEnum.Two


def test_implicit_conversion():
    assert str(m.ClassWithUnscopedEnum.EMode.EFirstMode) == "EMode.EFirstMode"
    assert str(m.ClassWithUnscopedEnum.EFirstMode) == "EMode.EFirstMode"

    f = m.ClassWithUnscopedEnum.test_function
    first = m.ClassWithUnscopedEnum.EFirstMode
    second = m.ClassWithUnscopedEnum.ESecondMode

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
    assert int(m.Flags.Read) == 4
    assert int(m.Flags.Write) == 2
    assert int(m.Flags.Execute) == 1
    assert int(m.Flags.Read | m.Flags.Write | m.Flags.Execute) == 7
    assert int(m.Flags.Read | m.Flags.Write) == 6
    assert int(m.Flags.Read | m.Flags.Execute) == 5
    assert int(m.Flags.Write | m.Flags.Execute) == 3
    assert int(m.Flags.Write | 1) == 3

    state = m.Flags.Read | m.Flags.Write
    assert (state & m.Flags.Read) != 0
    assert (state & m.Flags.Write) != 0
    assert (state & m.Flags.Execute) == 0
    assert (state & 1) == 0

    state2 = ~state
    assert state2 == -7
    assert int(state ^ state2) == -1


def test_enum_to_int():
    m.test_enum_to_int(m.Flags.Read)
    m.test_enum_to_int(m.ClassWithUnscopedEnum.EMode.EFirstMode)
    m.test_enum_to_uint(m.Flags.Read)
    m.test_enum_to_uint(m.ClassWithUnscopedEnum.EMode.EFirstMode)
    m.test_enum_to_long_long(m.Flags.Read)
    m.test_enum_to_long_long(m.ClassWithUnscopedEnum.EMode.EFirstMode)
