# -*- coding: utf-8 -*-

from pybind11_tests import classh_inheritance as m


def test_make_drvd_pass_base():
    d = m.make_drvd()
    i = m.pass_base(d)  # load_impl Case 2a
    assert i == 2 * 100 + 11


def test_make_drvd_up_cast_pass_drvd():
    b = m.make_drvd_up_cast()
    # the base return is down-cast immediately.
    assert b.__class__.__name__ == "drvd"
    i = m.pass_drvd(b)
    assert i == 2 * 100 + 12


def test_make_drvd2_pass_bases():
    d = m.make_drvd2()
    i1 = m.pass_base1(d)  # load_impl Case 2c
    assert i1 == 3 * 110 + 4 * 120 + 21
    i2 = m.pass_base2(d)
    assert i2 == 3 * 110 + 4 * 120 + 22


def test_make_drvd2_up_casts_pass_drvd2():
    b1 = m.make_drvd2_up_cast1()
    assert b1.__class__.__name__ == "drvd2"
    i1 = m.pass_drvd2(b1)
    assert i1 == 3 * 110 + 4 * 120 + 23
    b2 = m.make_drvd2_up_cast2()
    assert b2.__class__.__name__ == "drvd2"
    i2 = m.pass_drvd2(b2)
    assert i2 == 3 * 110 + 4 * 120 + 23
