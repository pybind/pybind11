# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import class_sh_disowning as m


def test_same_twice():
    while True:
        obj1a = m.Atype1(57)
        obj1b = m.Atype1(62)
        assert m.same_twice(obj1a, obj1b) == (57 * 10 + 1) * 100 + (62 * 10 + 1) * 10
        obj1c = m.Atype1(0)
        with pytest.raises(ValueError):
            m.same_twice(obj1c, obj1c)  # 1st disowning works, 2nd fails.
        with pytest.raises(ValueError):
            obj1c.get()
        return  # Comment out for manual leak checking (use `top` command).


def test_mixed():
    while True:
        obj1a = m.Atype1(90)
        obj2a = m.Atype2(25)
        assert m.mixed(obj1a, obj2a) == (90 * 10 + 1) * 200 + (25 * 10 + 2) * 20
        obj1b = m.Atype1(0)
        with pytest.raises(ValueError):
            m.mixed(obj1b, obj2a)
        with pytest.raises(ValueError):
            obj1b.get()  # obj1b was disowned even though m.mixed(obj1b, obj2a) failed.
        return  # Comment out for manual leak checking (use `top` command).


def test_overloaded():
    while True:
        obj1 = m.Atype1(81)
        obj2 = m.Atype2(60)
        with pytest.raises(TypeError):
            m.overloaded(obj1, "NotInt")
        assert obj1.get() == 81 * 10 + 1  # Not disowned.
        assert m.overloaded(obj1, 3) == (81 * 10 + 1) * 30 + 3
        with pytest.raises(TypeError):
            m.overloaded(obj2, "NotInt")
        assert obj2.get() == 60 * 10 + 2  # Not disowned.
        assert m.overloaded(obj2, 2) == (60 * 10 + 2) * 40 + 2
        return  # Comment out for manual leak checking (use `top` command).
