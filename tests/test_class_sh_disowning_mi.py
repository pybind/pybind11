# -*- coding: utf-8 -*-

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


def was_disowned(obj):
    try:
        obj.get()
    except ValueError as e:
        assert (
            str(e)
            == "Missing value for wrapped C++ type: Python instance was disowned."
        )
        return True
    return False


def test_disown_b():
    b = m.B()
    assert b.get() == 10
    m.disown(b)
    assert was_disowned(b)


def test_disown_c0():
    c0 = m.C0()
    assert c0.get() == 1020
    b = c0.b()
    m.disown(c0)
    assert was_disowned(c0)
    assert was_disowned(b)


def test_disown_c1():
    c1 = m.C1()
    assert c1.get() == 1021
    b = c1.b()
    m.disown(c1)
    assert was_disowned(c1)
    assert was_disowned(b)


def test_disown_d():
    d = m.D()
    assert d.get() == 10202130
    b = d.b()
    c0 = d.c0()
    c1 = d.c1()
    m.disown(d)
    assert was_disowned(d)
    assert was_disowned(c1)
    assert was_disowned(c0)
    assert was_disowned(b)
