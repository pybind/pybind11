# Adapted from:
# https://github.com/google/clif/blob/5718e4d0807fd3b6a8187dde140069120b81ecef/clif/testing/python/python_multiple_inheritance_test.py

from pybind11_tests import python_multiple_inheritance as m


class PC(m.CppBase):
    pass


class PPCCInit(PC, m.CppDrvd):
    def __init__(self, value):
        print("\nLOOOK PPCCInit PC", flush=True)
        PC.__init__(self, value)
        print("\nLOOOK PPCCInit CppDrvd", flush=True)
        m.CppDrvd.__init__(self, value + 1)
        print("\nLOOOK PPCCInit Done", flush=True)


def NOtest_PC_AAA():
    print("\nLOOOK BEFORE PC(11) AAA", flush=True)
    d = PC(11)
    print("\nLOOOK  AFTER PC(11) AAA", flush=True)
    assert d.get_base_value() == 11
    d.reset_base_value(13)
    assert d.get_base_value() == 13


# Moving this test after test_PC() changes the behavior!
def test_PPCCInit_BBB():
    print("\nLOOOK BEFORE PPCCInit(11) BBB", flush=True)
    d = PPCCInit(11)
    print("\nLOOOK  AFTER PPCCInit(11) BBB", flush=True)
    print("\nLOOOK", flush=True)
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


def NOtest_PC_CCC():
    print("\nLOOOK BEFORE PC(11) CCC", flush=True)
    d = PC(11)
    print("\nLOOOK  AFTER PC(11) CCC", flush=True)
    assert d.get_base_value() == 11
    d.reset_base_value(13)
    assert d.get_base_value() == 13

# Moving this test after test_PC() changes the behavior!
def NOtest_PPCCInit_DDD():
    print("\nLOOOK BEFORE PPCCInit(11) DDD", flush=True)
    d = PPCCInit(11)
    print("\nLOOOK  AFTER PPCCInit(11) DDD", flush=True)
    print("\nLOOOK", flush=True)
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
