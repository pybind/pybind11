import pytest

from pybind11_tests import class_sh_trampoline_basic as m


class PyDrvd0(m.Abase0):
    def __init__(self, val):
        super().__init__(val)

    def Add(self, other_val):
        return self.Get() * 100 + other_val


class PyDrvd1(m.Abase1):
    def __init__(self, val):
        super().__init__(val)

    def Add(self, other_val):
        return self.Get() * 200 + other_val


def test_drvd0_add():
    drvd = PyDrvd0(74)
    assert drvd.Add(38) == (74 * 10 + 3) * 100 + 38


def test_drvd0_add_in_cpp_raw_ptr():
    drvd = PyDrvd0(52)
    assert m.AddInCppRawPtr(drvd, 27) == ((52 * 10 + 3) * 100 + 27) * 10 + 7


def test_drvd0_add_in_cpp_shared_ptr():
    while True:
        drvd = PyDrvd0(36)
        assert m.AddInCppSharedPtr(drvd, 56) == ((36 * 10 + 3) * 100 + 56) * 100 + 11
        return  # Comment out for manual leak checking (use `top` command).


def test_drvd0_add_in_cpp_unique_ptr():
    while True:
        drvd = PyDrvd0(0)
        with pytest.raises(ValueError) as exc_info:
            m.AddInCppUniquePtr(drvd, 0)
        assert (
            str(exc_info.value)
            == "Alias class (also known as trampoline) does not inherit from"
            " py::trampoline_self_life_support, therefore the ownership of this"
            " instance cannot safely be transferred to C++."
        )
        return  # Comment out for manual leak checking (use `top` command).


def test_drvd1_add_in_cpp_unique_ptr():
    while True:
        drvd = PyDrvd1(25)
        assert m.AddInCppUniquePtr(drvd, 83) == ((25 * 10 + 3) * 200 + 83) * 100 + 13
        return  # Comment out for manual leak checking (use `top` command).
