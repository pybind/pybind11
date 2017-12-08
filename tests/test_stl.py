import pytest

from pybind11_tests import stl as m
from pybind11_tests import UserType


def test_vector(doc):
    """std::vector <-> list"""
    lst = m.cast_vector()
    assert lst == [1]
    lst.append(2)
    assert m.load_vector(lst)
    assert m.load_vector(tuple(lst))

    assert m.cast_bool_vector() == [True, False]
    assert m.load_bool_vector([True, False])

    assert doc(m.cast_vector) == "cast_vector() -> List[int]"
    assert doc(m.load_vector) == "load_vector(arg0: List[int]) -> bool"

    # Test regression caused by 936: pointers to stl containers weren't castable
    assert m.cast_ptr_vector() == ["lvalue", "lvalue"]


def test_array(doc):
    """std::array <-> list"""
    lst = m.cast_array()
    assert lst == [1, 2]
    assert m.load_array(lst)

    assert doc(m.cast_array) == "cast_array() -> List[int[2]]"
    assert doc(m.load_array) == "load_array(arg0: List[int[2]]) -> bool"


def test_valarray(doc):
    """std::valarray <-> list"""
    lst = m.cast_valarray()
    assert lst == [1, 4, 9]
    assert m.load_valarray(lst)

    assert doc(m.cast_valarray) == "cast_valarray() -> List[int]"
    assert doc(m.load_valarray) == "load_valarray(arg0: List[int]) -> bool"


def test_map(doc):
    """std::map <-> dict"""
    d = m.cast_map()
    assert d == {"key": "value"}
    d["key2"] = "value2"
    assert m.load_map(d)

    assert doc(m.cast_map) == "cast_map() -> Dict[str, str]"
    assert doc(m.load_map) == "load_map(arg0: Dict[str, str]) -> bool"


def test_set(doc):
    """std::set <-> set"""
    s = m.cast_set()
    assert s == {"key1", "key2"}
    s.add("key3")
    assert m.load_set(s)

    assert doc(m.cast_set) == "cast_set() -> Set[str]"
    assert doc(m.load_set) == "load_set(arg0: Set[str]) -> bool"


def test_recursive_casting():
    """Tests that stl casters preserve lvalue/rvalue context for container values"""
    assert m.cast_rv_vector() == ["rvalue", "rvalue"]
    assert m.cast_lv_vector() == ["lvalue", "lvalue"]
    assert m.cast_rv_array() == ["rvalue", "rvalue", "rvalue"]
    assert m.cast_lv_array() == ["lvalue", "lvalue"]
    assert m.cast_rv_map() == {"a": "rvalue"}
    assert m.cast_lv_map() == {"a": "lvalue", "b": "lvalue"}
    assert m.cast_rv_nested() == [[[{"b": "rvalue", "c": "rvalue"}], [{"a": "rvalue"}]]]
    assert m.cast_lv_nested() == {
        "a": [[["lvalue", "lvalue"]], [["lvalue", "lvalue"]]],
        "b": [[["lvalue", "lvalue"], ["lvalue", "lvalue"]]]
    }

    # Issue #853 test case:
    z = m.cast_unique_ptr_vector()
    assert z[0].value == 7 and z[1].value == 42


def test_move_out_container():
    """Properties use the `reference_internal` policy by default. If the underlying function
    returns an rvalue, the policy is automatically changed to `move` to avoid referencing
    a temporary. In case the return value is a container of user-defined types, the policy
    also needs to be applied to the elements, not just the container."""
    c = m.MoveOutContainer()
    moved_out_list = c.move_list
    assert [x.value for x in moved_out_list] == [0, 1, 2]


def test_vec_of_reference_wrapper():
    """#171: Can't return reference wrappers (or STL structures containing them)"""
    assert str(m.return_vec_of_reference_wrapper(UserType(4))) == \
        "[UserType(1), UserType(2), UserType(3), UserType(4)]"


def test_stl_pass_by_pointer(msg):
    """Passing nullptr or None to an STL container pointer is not expected to work"""
    with pytest.raises(TypeError) as excinfo:
        m.stl_pass_by_pointer()  # default value is `nullptr`
    assert msg(excinfo.value) == """
        stl_pass_by_pointer(): incompatible function arguments. The following argument types are supported:
            1. (v: List[int]=None) -> List[int]

        Invoked with:
    """  # noqa: E501 line too long

    with pytest.raises(TypeError) as excinfo:
        m.stl_pass_by_pointer(None)
    assert msg(excinfo.value) == """
        stl_pass_by_pointer(): incompatible function arguments. The following argument types are supported:
            1. (v: List[int]=None) -> List[int]

        Invoked with: None
    """  # noqa: E501 line too long

    assert m.stl_pass_by_pointer([1, 2, 3]) == [1, 2, 3]


def test_missing_header_message():
    """Trying convert `list` to a `std::vector`, or vice versa, without including
    <pybind11/stl.h> should result in a helpful suggestion in the error message"""
    import pybind11_cross_module_tests as cm

    expected_message = (
        "Did you forget to `#include <pybind11/stl.h>`? Or <pybind11/complex.h>,\n"
        "<pybind11/functional.h>, <pybind11/chrono.h>, <pybind11/utility.h>, etc.\n"
        "Some automatic conversions are optional and require extra headers to be\n"
        "included when compiling your pybind11 module.")

    with pytest.raises(TypeError) as excinfo:
        cm.missing_header_arg([1.0, 2.0, 3.0])
    assert expected_message in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        cm.missing_header_return()
    assert expected_message in str(excinfo.value)
