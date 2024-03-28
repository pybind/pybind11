# Adapted from:
# https://github.com/google/clif/blob/5718e4d0807fd3b6a8187dde140069120b81ecef/clif/testing/python/python_multiple_inheritance_test.py

import pytest

from pybind11_tests import python_multiple_inheritance as m


class PC(m.CppBase):
    pass


class PPCC(PC, m.CppDrvd):
    pass


class PPPCCC(PPCC, m.CppDrvd2):
    pass


class PC1(m.CppDrvd):
    pass


class PC2(m.CppDrvd2):
    pass


class PCD(PC1, PC2):
    pass


class PCDI(PC1, PC2):
    def __init__(self):
        PC1.__init__(self, 11)
        PC2.__init__(self, 12)


def test_PC():
    d = PC(11)
    assert d.get_base_value() == 11
    d.reset_base_value(13)
    assert d.get_base_value() == 13


def test_PPCC():
    d = PPCC(11)
    assert d.get_drvd_value() == 33
    d.reset_drvd_value(55)
    assert d.get_drvd_value() == 55

    assert d.get_base_value() == 11
    assert d.get_base_value_from_drvd() == 11
    d.reset_base_value(20)
    assert d.get_base_value() == 20
    assert d.get_base_value_from_drvd() == 20
    d.reset_base_value_from_drvd(30)
    assert d.get_base_value() == 30
    assert d.get_base_value_from_drvd() == 30


def NOtest_PPPCCC():
    # terminate called after throwing an instance of 'pybind11::error_already_set'
    # what():  TypeError: bases include diverging derived types:
    #     base=pybind11_tests.python_multiple_inheritance.CppBase,
    #     derived1=pybind11_tests.python_multiple_inheritance.CppDrvd,
    #     derived2=pybind11_tests.python_multiple_inheritance.CppDrvd2
    PPPCCC(11)


def test_PCD():
    # This escapes all_type_info_check_for_divergence() because CppBase does not appear in bases.
    with pytest.raises(
        TypeError,
        match=r"CppDrvd2\.__init__\(\) must be called when overriding __init__$",
    ):
        PCD(11)


def test_PCDI():
    obj = PCDI()
    with pytest.raises(TypeError, match="^bases include diverging derived types: "):
        m.pass_CppBase(obj)
