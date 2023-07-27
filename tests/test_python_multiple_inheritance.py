# Adapted from:
# https://github.com/google/clif/blob/5718e4d0807fd3b6a8187dde140069120b81ecef/clif/testing/python/python_multiple_inheritance_test.py

from pybind11_tests import python_multiple_inheritance as m


class PC(m.CppBase):
    pass


class PPCCInit(PC, m.CppDrvd):
    def __init__(self, value):
        PC.__init__(self, value)
        m.CppDrvd.__init__(self, value + 1)


# Moving this test after test_PC() changes the behavior!
def test_PPCCInit():
    d = PPCCInit(11)
    assert d.get_drvd_value() == 36
    d.reset_drvd_value(55)
    assert d.get_drvd_value() == 55

    assert d.get_base_value() == 12
    assert d.get_base_value_from_drvd() == 12
    d.reset_base_value(20)
    assert d.get_base_value() == 20
    assert d.get_base_value_from_drvd() == 20
    d.reset_base_value_from_drvd(30)
    assert d.get_base_value() == 30
    assert d.get_base_value_from_drvd() == 30


def test_PC():
    d = PC(11)
    assert d.get_base_value() == 11
    d.reset_base_value(13)
    assert d.get_base_value() == 13
