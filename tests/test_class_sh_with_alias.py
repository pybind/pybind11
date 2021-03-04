# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import class_sh_with_alias as m


class PyDrvd(m.Abase):
    def __init__(self, val):
        super(PyDrvd, self).__init__(val)

    def Add(self, other_val):  # noqa:  N802
        return self.Get() * 100 + other_val


def test_drvd_add():
    drvd = PyDrvd(74)
    assert drvd.Add(38) == (74 * 10 + 3) * 100 + 38


def test_add_in_cpp_raw_ptr():
    drvd = PyDrvd(52)
    assert m.AddInCppRawPtr(drvd, 27) == ((52 * 10 + 3) * 100 + 27) * 10 + 7


def test_add_in_cpp_shared_ptr():
    drvd = PyDrvd(36)
    assert m.AddInCppSharedPtr(drvd, 56) == ((36 * 10 + 3) * 100 + 56) * 100 + 11


def test_add_in_cpp_unique_ptr():
    drvd = PyDrvd(0)
    with pytest.raises(ValueError) as exc_info:
        m.AddInCppUniquePtr(drvd, 0)
    assert (
        str(exc_info.value)
        == "Ownership of instance with virtual overrides in Python cannot be transferred to C++."
    )
