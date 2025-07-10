from __future__ import annotations

import pytest

from pybind11_tests import class_sh_virtual_py_cpp_mix as m


class PyBase(m.Base):  # Avoiding name PyDerived, for more systematic naming.
    def __init__(self):
        m.Base.__init__(self)

    def get(self):
        return 323


class PyCppDerived(m.CppDerived):
    def __init__(self):
        m.CppDerived.__init__(self)

    def get(self):
        return 434


@pytest.mark.parametrize(
    ("ctor", "expected"),
    [
        (m.Base, 101),
        (PyBase, 323),
        (m.CppDerivedPlain, 202),
        (m.CppDerived, 212),
        (PyCppDerived, 434),
    ],
)
def test_base_get(ctor, expected):
    obj = ctor()
    assert obj.get() == expected


@pytest.mark.parametrize(
    ("ctor", "expected"),
    [
        (m.Base, 4101),
        (PyBase, 4323),
        (m.CppDerivedPlain, 4202),
        (m.CppDerived, 4212),
        (PyCppDerived, 4434),
    ],
)
def test_get_from_cpp_plainc_ptr(ctor, expected):
    obj = ctor()
    assert m.get_from_cpp_plainc_ptr(obj) == expected


@pytest.mark.parametrize(
    ("ctor", "expected"),
    [
        (m.Base, 5101),
        (PyBase, 5323),
        (m.CppDerivedPlain, 5202),
        (m.CppDerived, 5212),
        (PyCppDerived, 5434),
    ],
)
def test_get_from_cpp_unique_ptr(ctor, expected):
    obj = ctor()
    assert m.get_from_cpp_unique_ptr(obj) == expected
