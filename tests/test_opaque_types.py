from __future__ import annotations

import pytest

import env
from pybind11_tests import ConstructorStats, UserType
from pybind11_tests import opaque_types as m


def test_string_list():
    lst = m.StringList()
    lst.push_back("Element 1")
    lst.push_back("Element 2")
    assert m.print_opaque_list(lst) == "Opaque list: [Element 1, Element 2]"
    assert lst.back() == "Element 2"

    for i, k in enumerate(lst, start=1):
        assert k == f"Element {i}"
    lst.pop_back()
    assert m.print_opaque_list(lst) == "Opaque list: [Element 1]"

    cvp = m.ClassWithSTLVecProperty()
    assert m.print_opaque_list(cvp.stringList) == "Opaque list: []"

    cvp.stringList = lst
    cvp.stringList.push_back("Element 3")
    assert m.print_opaque_list(cvp.stringList) == "Opaque list: [Element 1, Element 3]"


def test_pointers(msg, backport_typehints):
    living_before = ConstructorStats.get(UserType).alive()
    assert m.get_void_ptr_value(m.return_void_ptr()) == 0x1234
    assert m.get_void_ptr_value(UserType())  # Should also work for other C++ types

    if not env.GRAALPY:
        assert ConstructorStats.get(UserType).alive() == living_before

    with pytest.raises(TypeError) as excinfo:
        m.get_void_ptr_value([1, 2, 3])  # This should not work

    assert (
        backport_typehints(msg(excinfo.value))
        == """
            get_void_ptr_value(): incompatible function arguments. The following argument types are supported:
                1. (arg0: types.CapsuleType) -> int

            Invoked with: [1, 2, 3]
        """
    )

    assert m.return_null_str() is None
    assert m.get_null_str_value(m.return_null_str()) is not None

    ptr = m.return_unique_ptr()
    assert "StringList" in repr(ptr)
    assert m.print_opaque_list(ptr) == "Opaque list: [some value]"


def test_unions():
    int_float_union = m.IntFloat()
    int_float_union.i = 42
    assert int_float_union.i == 42
    int_float_union.f = 3.0
    assert int_float_union.f == 3.0


def test_issue_5988_opaque_std_array():
    """Regression test for GitHub issue #5988: crash when binding with opaque std::array types."""
    # Test basic Array3d (opaque std::array<double, 3>) functionality
    a = m.Array3d()
    a[0] = 1.0
    a[1] = 2.5
    a[2] = 3.0
    assert a[0] == 1.0
    assert a[1] == 2.5
    assert a[2] == 3.0
    assert len(a) == 3

    with pytest.raises(IndexError):
        _ = a[3]

    # Test VecArray3d (opaque std::vector<std::array<double, 3>>) functionality
    v = m.VecArray3d()
    assert len(v) == 0
    v.push_back(a)
    assert len(v) == 1
    assert v[0][0] == 1.0
    assert v[0][1] == 2.5
    assert v[0][2] == 3.0

    with pytest.raises(IndexError):
        _ = v[1]
