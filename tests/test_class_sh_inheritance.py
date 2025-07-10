from __future__ import annotations

from pybind11_tests import class_sh_inheritance as m


def test_rtrn_mptr_drvd_pass_cptr_base():
    d = m.rtrn_mptr_drvd()
    i = m.pass_cptr_base(d)  # load_impl Case 2a
    assert i == 2 * 100 + 11


def test_rtrn_shmp_drvd_pass_shcp_base():
    d = m.rtrn_shmp_drvd()
    i = m.pass_shcp_base(d)  # load_impl Case 2a
    assert i == 2 * 100 + 21


def test_rtrn_mptr_drvd_up_cast_pass_cptr_drvd():
    b = m.rtrn_mptr_drvd_up_cast()
    # the base return is down-cast immediately.
    assert b.__class__.__name__ == "drvd"
    i = m.pass_cptr_drvd(b)
    assert i == 2 * 100 + 12


def test_rtrn_shmp_drvd_up_cast_pass_shcp_drvd():
    b = m.rtrn_shmp_drvd_up_cast()
    # the base return is down-cast immediately.
    assert b.__class__.__name__ == "drvd"
    i = m.pass_shcp_drvd(b)
    assert i == 2 * 100 + 22


def test_rtrn_mptr_drvd2_pass_cptr_bases():
    d = m.rtrn_mptr_drvd2()
    i1 = m.pass_cptr_base1(d)  # load_impl Case 2c
    assert i1 == 3 * 110 + 4 * 120 + 21
    i2 = m.pass_cptr_base2(d)
    assert i2 == 3 * 110 + 4 * 120 + 22


def test_rtrn_mptr_drvd2_up_casts_pass_cptr_drvd2():
    b1 = m.rtrn_mptr_drvd2_up_cast1()
    assert b1.__class__.__name__ == "drvd2"
    i1 = m.pass_cptr_drvd2(b1)
    assert i1 == 3 * 110 + 4 * 120 + 23
    b2 = m.rtrn_mptr_drvd2_up_cast2()
    assert b2.__class__.__name__ == "drvd2"
    i2 = m.pass_cptr_drvd2(b2)
    assert i2 == 3 * 110 + 4 * 120 + 23


def test_python_drvd2():
    class Drvd2(m.base1, m.base2):
        def __init__(self):
            m.base1.__init__(self)
            m.base2.__init__(self)

    d = Drvd2()
    i1 = m.pass_cptr_base1(d)  # load_impl Case 2b
    assert i1 == 110 + 21
    i2 = m.pass_cptr_base2(d)
    assert i2 == 120 + 22
