from __future__ import annotations

from pybind11_tests import class_sh_trampoline_basic as m


class PyDrvd(m.Abase):
    def __init__(self, val):
        super().__init__(val)

    def Add(self, other_val):
        return self.Get() * 100 + other_val


def test_drvd_add():
    drvd = PyDrvd(74)
    assert drvd.Add(38) == (74 * 10 + 3) * 100 + 38


def test_drvd_add_in_cpp_raw_ptr():
    drvd = PyDrvd(52)
    assert m.AddInCppRawPtr(drvd, 27) == ((52 * 10 + 3) * 100 + 27) * 10 + 7


def test_drvd_add_in_cpp_shared_ptr():
    while True:
        drvd = PyDrvd(36)
        assert m.AddInCppSharedPtr(drvd, 56) == ((36 * 10 + 3) * 100 + 56) * 100 + 11
        return  # Comment out for manual leak checking (use `top` command).


def test_drvd_add_in_cpp_unique_ptr():
    while True:
        drvd = PyDrvd(25)
        assert m.AddInCppUniquePtr(drvd, 83) == ((25 * 10 + 3) * 100 + 83) * 100 + 13
        return  # Comment out for manual leak checking (use `top` command).
