import pytest

import env  # noqa: F401
from pybind11_tests import class_sh_disowning_mi as m


def test_diamond_inheritance():
    # Very similar to test_multiple_inheritance.py:test_diamond_inheritance.
    d = m.D()
    assert d is d.d()
    assert d is d.c0()
    assert d is d.c1()
    assert d is d.b()
    assert d is d.c0().b()
    assert d is d.c1().b()
    assert d is d.c0().c1().b().c0().b()


def is_disowned(callable_method):
    try:
        callable_method()
    except ValueError as e:
        assert "Python instance was disowned" in str(e)  # noqa: PT017
        return True
    return False


def test_disown_b():
    b = m.B()
    assert b.get() == 10
    m.disown_b(b)
    assert is_disowned(b.get)


@pytest.mark.parametrize("var_to_disown", ["c0", "b"])
def test_disown_c0(var_to_disown):
    c0 = m.C0()
    assert c0.get() == 1020
    b = c0.b()
    m.disown_b(locals()[var_to_disown])
    assert is_disowned(c0.get)
    assert is_disowned(b.get)


@pytest.mark.parametrize("var_to_disown", ["c1", "b"])
def test_disown_c1(var_to_disown):
    c1 = m.C1()
    assert c1.get() == 1021
    b = c1.b()
    m.disown_b(locals()[var_to_disown])
    assert is_disowned(c1.get)
    assert is_disowned(b.get)


@pytest.mark.parametrize("var_to_disown", ["d", "c1", "c0", "b"])
def test_disown_d(var_to_disown):
    d = m.D()
    assert d.get() == 10202130
    b = d.b()
    c0 = d.c0()
    c1 = d.c1()
    m.disown_b(locals()[var_to_disown])
    assert is_disowned(d.get)
    assert is_disowned(c1.get)
    assert is_disowned(c0.get)
    assert is_disowned(b.get)


# Based on test_multiple_inheritance.py:test_multiple_inheritance_python.
class MI1(m.Base1, m.Base2):
    def __init__(self, i, j):
        m.Base1.__init__(self, i)
        m.Base2.__init__(self, j)


class B1:
    def v(self):
        return 1


class MI2(B1, m.Base1, m.Base2):
    def __init__(self, i, j):
        B1.__init__(self)
        m.Base1.__init__(self, i)
        m.Base2.__init__(self, j)


class MI3(MI2):
    def __init__(self, i, j):
        MI2.__init__(self, i, j)


class MI4(MI3, m.Base2):
    def __init__(self, i, j):
        MI3.__init__(self, i, j)
        # This should be ignored (Base2 is already initialized via MI2):
        m.Base2.__init__(self, i + 100)


class MI5(m.Base2, B1, m.Base1):
    def __init__(self, i, j):
        B1.__init__(self)
        m.Base1.__init__(self, i)
        m.Base2.__init__(self, j)


class MI6(m.Base2, B1):
    def __init__(self, i):
        m.Base2.__init__(self, i)
        B1.__init__(self)


class B2(B1):
    def v(self):
        return 2


class B3:
    def v(self):
        return 3


class B4(B3, B2):
    def v(self):
        return 4


class MI7(B4, MI6):
    def __init__(self, i):
        B4.__init__(self)
        MI6.__init__(self, i)


class MI8(MI6, B3):
    def __init__(self, i):
        MI6.__init__(self, i)
        B3.__init__(self)


class MI8b(B3, MI6):
    def __init__(self, i):
        B3.__init__(self)
        MI6.__init__(self, i)


@pytest.mark.xfail("env.PYPY")
def test_multiple_inheritance_python():
    # Based on test_multiple_inheritance.py:test_multiple_inheritance_python.
    # Exercises values_and_holders with 2 value_and_holder instances.

    mi1 = MI1(1, 2)
    assert mi1.foo() == 1
    assert mi1.bar() == 2

    mi2 = MI2(3, 4)
    assert mi2.v() == 1
    assert mi2.foo() == 3
    assert mi2.bar() == 4

    mi3 = MI3(5, 6)
    assert mi3.v() == 1
    assert mi3.foo() == 5
    assert mi3.bar() == 6

    mi4 = MI4(7, 8)
    assert mi4.v() == 1
    assert mi4.foo() == 7
    assert mi4.bar() == 8

    mi5 = MI5(10, 11)
    assert mi5.v() == 1
    assert mi5.foo() == 10
    assert mi5.bar() == 11

    mi6 = MI6(12)
    assert mi6.v() == 1
    assert mi6.bar() == 12

    mi7 = MI7(13)
    assert mi7.v() == 4
    assert mi7.bar() == 13

    mi8 = MI8(14)
    assert mi8.v() == 1
    assert mi8.bar() == 14

    mi8b = MI8b(15)
    assert mi8b.v() == 3
    assert mi8b.bar() == 15


DISOWN_CLS_I_J_V_LIST = [
    (MI1, 1, 2, None),
    (MI2, 3, 4, 1),
    (MI3, 5, 6, 1),
    (MI4, 7, 8, 1),
    (MI5, 10, 11, 1),
]


@pytest.mark.xfail("env.PYPY", strict=False)
@pytest.mark.parametrize(("cls", "i", "j", "v"), DISOWN_CLS_I_J_V_LIST)
def test_disown_base1_first(cls, i, j, v):
    obj = cls(i, j)
    assert obj.foo() == i
    assert m.disown_base1(obj) == 2000 * i + 1
    assert is_disowned(obj.foo)
    assert obj.bar() == j
    assert m.disown_base2(obj) == 2000 * j + 2
    assert is_disowned(obj.bar)
    if v is not None:
        assert obj.v() == v


@pytest.mark.xfail("env.PYPY", strict=False)
@pytest.mark.parametrize(("cls", "i", "j", "v"), DISOWN_CLS_I_J_V_LIST)
def test_disown_base2_first(cls, i, j, v):
    obj = cls(i, j)
    assert obj.bar() == j
    assert m.disown_base2(obj) == 2000 * j + 2
    assert is_disowned(obj.bar)
    assert obj.foo() == i
    assert m.disown_base1(obj) == 2000 * i + 1
    assert is_disowned(obj.foo)
    if v is not None:
        assert obj.v() == v


@pytest.mark.xfail("env.PYPY", strict=False)
@pytest.mark.parametrize(
    ("cls", "j", "v"),
    [
        (MI6, 12, 1),
        (MI7, 13, 4),
        (MI8, 14, 1),
        (MI8b, 15, 3),
    ],
)
def test_disown_base2(cls, j, v):
    obj = cls(j)
    assert obj.bar() == j
    assert m.disown_base2(obj) == 2000 * j + 2
    assert is_disowned(obj.bar)
    assert obj.v() == v
