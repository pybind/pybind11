# -*- coding: utf-8 -*-

from pybind11_tests import class_sh_virtual_py_cpp_mix as m


class PyDerived(m.Base):
    def __init__(self):
        m.Base.__init__(self)

    def get(self):
        return 323


def test_py_derived_get():
    d = PyDerived()
    assert d.get() == 323


def test_get_from_cpp_plainc_ptr_passing_py_derived():
    d = PyDerived()
    assert m.get_from_cpp_plainc_ptr(d) == 4323


def test_get_from_cpp_unique_ptr_passing_py_derived():
    d = PyDerived()
    assert m.get_from_cpp_unique_ptr(d) == 5323


def test_cpp_derived_get():
    d = m.CppDerived()
    assert d.get() == 212


def test_get_from_cpp_plainc_ptr_passing_cpp_derived():
    d = m.CppDerived()
    assert m.get_from_cpp_plainc_ptr(d) == 4212


def test_get_from_cpp_unique_ptr_passing_cpp_derived():
    d = m.CppDerived()
    assert m.get_from_cpp_unique_ptr(d) == 5212
