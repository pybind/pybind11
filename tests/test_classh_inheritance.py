# -*- coding: utf-8 -*-

from pybind11_tests import classh_inheritance as m


def test_make_drvd_pass_base():
    d = m.make_drvd()
    i = m.pass_base(d)
    assert i == 200


def test_make_drvd_up_cast_pass_drvd():
    b = m.make_drvd_up_cast()
    # the base return is down-cast immediately.
    assert b.__class__.__name__ == "drvd"
    i = m.pass_drvd(b)
    assert i == 200
