import pytest


def test_string_list(capture):
    from pybind11_tests import StringList, ClassWithSTLVecProperty, print_opaque_list

    l = StringList()
    l.push_back("Element 1")
    l.push_back("Element 2")
    with capture:
        print_opaque_list(l)
    assert capture == "Opaque list: [Element 1, Element 2]"
    assert l.back() == "Element 2"

    for i, k in enumerate(l, start=1):
        assert k == "Element {}".format(i)
    l.pop_back()
    with capture:
        print_opaque_list(l)
    assert capture == "Opaque list: [Element 1]"

    cvp = ClassWithSTLVecProperty()
    with capture:
        print_opaque_list(cvp.stringList)
    assert capture == "Opaque list: []"

    cvp.stringList = l
    cvp.stringList.push_back("Element 3")
    with capture:
        print_opaque_list(cvp.stringList)
    assert capture == "Opaque list: [Element 1, Element 3]"


def test_pointers(capture, msg):
    from pybind11_tests import (return_void_ptr, print_void_ptr, ExampleMandA,
                                print_opaque_list, return_null_str, print_null_str,
                                return_unique_ptr, ConstructorStats)

    with capture:
        print_void_ptr(return_void_ptr())
    assert capture == "Got void ptr : 0x1234"
    with capture:
        print_void_ptr(ExampleMandA())  # Should also work for other C++ types
    assert "Got void ptr" in capture
    assert ConstructorStats.get(ExampleMandA).alive() == 0

    with pytest.raises(TypeError) as excinfo:
        print_void_ptr([1, 2, 3])  # This should not work
    assert msg(excinfo.value) == """
        Incompatible function arguments. The following argument types are supported:
            1. (arg0: capsule) -> None
            Invoked with: [1, 2, 3]
    """

    assert return_null_str() is None
    with capture:
        print_null_str(return_null_str())
    assert capture == "Got null str : 0x0"

    ptr = return_unique_ptr()
    assert "StringList" in repr(ptr)
    with capture:
        print_opaque_list(ptr)
    assert capture == "Opaque list: [some value]"
