import pytest


def test_string_list():
    from pybind11_tests import StringList, ClassWithSTLVecProperty, print_opaque_list

    l = StringList()
    l.push_back("Element 1")
    l.push_back("Element 2")
    assert print_opaque_list(l) == "Opaque list: [Element 1, Element 2]"
    assert l.back() == "Element 2"

    for i, k in enumerate(l, start=1):
        assert k == "Element {}".format(i)
    l.pop_back()
    assert print_opaque_list(l) == "Opaque list: [Element 1]"

    cvp = ClassWithSTLVecProperty()
    assert print_opaque_list(cvp.stringList) == "Opaque list: []"

    cvp.stringList = l
    cvp.stringList.push_back("Element 3")
    assert print_opaque_list(cvp.stringList) == "Opaque list: [Element 1, Element 3]"


def test_pointers(msg):
    from pybind11_tests import (return_void_ptr, get_void_ptr_value, ExampleMandA,
                                print_opaque_list, return_null_str, get_null_str_value,
                                return_unique_ptr, ConstructorStats)

    assert get_void_ptr_value(return_void_ptr()) == 0x1234
    assert get_void_ptr_value(ExampleMandA())  # Should also work for other C++ types
    assert ConstructorStats.get(ExampleMandA).alive() == 0

    with pytest.raises(TypeError) as excinfo:
        get_void_ptr_value([1, 2, 3])  # This should not work
    assert msg(excinfo.value) == """
        get_void_ptr_value(): incompatible function arguments. The following argument types are supported:
            1. (arg0: capsule) -> int

        Invoked with: [1, 2, 3]
    """  # noqa: E501 line too long

    assert return_null_str() is None
    assert get_null_str_value(return_null_str()) is not None

    ptr = return_unique_ptr()
    assert "StringList" in repr(ptr)
    assert print_opaque_list(ptr) == "Opaque list: [some value]"
