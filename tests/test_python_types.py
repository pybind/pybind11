import pytest

from pybind11_tests import ExamplePythonTypes, ConstructorStats, has_optional, has_exp_optional


def test_repr():
    # In Python 3.3+, repr() accesses __qualname__
    assert "ExamplePythonTypes__Meta" in repr(type(ExamplePythonTypes))
    assert "ExamplePythonTypes" in repr(ExamplePythonTypes)


def test_static():
    ExamplePythonTypes.value = 15
    assert ExamplePythonTypes.value == 15
    assert ExamplePythonTypes.value2 == 5

    with pytest.raises(AttributeError) as excinfo:
        ExamplePythonTypes.value2 = 15
    assert str(excinfo.value) == "can't set attribute"


def test_instance(capture):
    with pytest.raises(TypeError) as excinfo:
        ExamplePythonTypes()
    assert str(excinfo.value) == "pybind11_tests.ExamplePythonTypes: No constructor defined!"

    instance = ExamplePythonTypes.new_instance()

    with capture:
        dict_result = instance.get_dict()
        dict_result['key2'] = 'value2'
        instance.print_dict(dict_result)
    assert capture.unordered == """
        key: key, value=value
        key: key2, value=value2
    """
    with capture:
        dict_result = instance.get_dict_2()
        dict_result['key2'] = 'value2'
        instance.print_dict_2(dict_result)
    assert capture.unordered == """
        key: key, value=value
        key: key2, value=value2
    """
    with capture:
        set_result = instance.get_set()
        set_result.add('key4')
        instance.print_set(set_result)
    assert capture.unordered == """
        key: key1
        key: key2
        key: key3
        key: key4
    """
    with capture:
        set_result = instance.get_set2()
        set_result.add('key3')
        instance.print_set_2(set_result)
    assert capture.unordered == """
        key: key1
        key: key2
        key: key3
    """
    with capture:
        list_result = instance.get_list()
        list_result.append('value2')
        instance.print_list(list_result)
    assert capture.unordered == """
        Entry at position 0: value
        list item 0: overwritten
        list item 1: value2
    """
    with capture:
        list_result = instance.get_list_2()
        list_result.append('value2')
        instance.print_list_2(list_result)
    assert capture.unordered == """
        list item 0: value
        list item 1: value2
    """
    with capture:
        list_result = instance.get_list_2()
        list_result.append('value2')
        instance.print_list_2(tuple(list_result))
    assert capture.unordered == """
        list item 0: value
        list item 1: value2
    """
    array_result = instance.get_array()
    assert array_result == ['array entry 1', 'array entry 2']
    with capture:
        instance.print_array(array_result)
    assert capture.unordered == """
        array item 0: array entry 1
        array item 1: array entry 2
    """
    varray_result = instance.get_valarray()
    assert varray_result == [1, 4, 9]
    with capture:
        instance.print_valarray(varray_result)
    assert capture.unordered == """
        valarray item 0: 1
        valarray item 1: 4
        valarray item 2: 9
    """
    with pytest.raises(RuntimeError) as excinfo:
        instance.throw_exception()
    assert str(excinfo.value) == "This exception was intentionally thrown."

    assert instance.pair_passthrough((True, "test")) == ("test", True)
    assert instance.tuple_passthrough((True, "test", 5)) == (5, "test", True)
    # Any sequence can be cast to a std::pair or std::tuple
    assert instance.pair_passthrough([True, "test"]) == ("test", True)
    assert instance.tuple_passthrough([True, "test", 5]) == (5, "test", True)

    assert instance.get_bytes_from_string().decode() == "foo"
    assert instance.get_bytes_from_str().decode() == "bar"
    assert instance.get_str_from_string().encode().decode() == "baz"
    assert instance.get_str_from_bytes().encode().decode() == "boo"

    class A(object):
        def __str__(self):
            return "this is a str"

        def __repr__(self):
            return "this is a repr"

    with capture:
        instance.test_print(A())
    assert capture == """
        this is a str
        this is a repr
    """

    cstats = ConstructorStats.get(ExamplePythonTypes)
    assert cstats.alive() == 1
    del instance
    assert cstats.alive() == 0


# PyPy does not seem to propagate the tp_docs field at the moment
def test_class_docs(doc):
    assert doc(ExamplePythonTypes) == "Example 2 documentation"


def test_method_docs(doc):
    assert doc(ExamplePythonTypes.get_dict) == """
        get_dict(self: m.ExamplePythonTypes) -> dict

        Return a Python dictionary
    """
    assert doc(ExamplePythonTypes.get_dict_2) == """
        get_dict_2(self: m.ExamplePythonTypes) -> Dict[str, str]

        Return a C++ dictionary
    """
    assert doc(ExamplePythonTypes.get_list) == """
        get_list(self: m.ExamplePythonTypes) -> list

        Return a Python list
    """
    assert doc(ExamplePythonTypes.get_list_2) == """
        get_list_2(self: m.ExamplePythonTypes) -> List[str]

        Return a C++ list
    """
    assert doc(ExamplePythonTypes.get_dict) == """
        get_dict(self: m.ExamplePythonTypes) -> dict

        Return a Python dictionary
    """
    assert doc(ExamplePythonTypes.get_set) == """
        get_set(self: m.ExamplePythonTypes) -> set

        Return a Python set
    """
    assert doc(ExamplePythonTypes.get_set2) == """
        get_set2(self: m.ExamplePythonTypes) -> Set[str]

        Return a C++ set
    """
    assert doc(ExamplePythonTypes.get_array) == """
        get_array(self: m.ExamplePythonTypes) -> List[str[2]]

        Return a C++ array
    """
    assert doc(ExamplePythonTypes.get_valarray) == """
        get_valarray(self: m.ExamplePythonTypes) -> List[int]

        Return a C++ valarray
    """
    assert doc(ExamplePythonTypes.print_dict) == """
        print_dict(self: m.ExamplePythonTypes, arg0: dict) -> None

        Print entries of a Python dictionary
    """
    assert doc(ExamplePythonTypes.print_dict_2) == """
        print_dict_2(self: m.ExamplePythonTypes, arg0: Dict[str, str]) -> None

        Print entries of a C++ dictionary
    """
    assert doc(ExamplePythonTypes.print_set) == """
        print_set(self: m.ExamplePythonTypes, arg0: set) -> None

        Print entries of a Python set
    """
    assert doc(ExamplePythonTypes.print_set_2) == """
        print_set_2(self: m.ExamplePythonTypes, arg0: Set[str]) -> None

        Print entries of a C++ set
    """
    assert doc(ExamplePythonTypes.print_list) == """
        print_list(self: m.ExamplePythonTypes, arg0: list) -> None

        Print entries of a Python list
    """
    assert doc(ExamplePythonTypes.print_list_2) == """
        print_list_2(self: m.ExamplePythonTypes, arg0: List[str]) -> None

        Print entries of a C++ list
    """
    assert doc(ExamplePythonTypes.print_array) == """
        print_array(self: m.ExamplePythonTypes, arg0: List[str[2]]) -> None

        Print entries of a C++ array
    """
    assert doc(ExamplePythonTypes.pair_passthrough) == """
        pair_passthrough(self: m.ExamplePythonTypes, arg0: Tuple[bool, str]) -> Tuple[str, bool]

        Return a pair in reversed order
    """
    assert doc(ExamplePythonTypes.tuple_passthrough) == """
        tuple_passthrough(self: m.ExamplePythonTypes, arg0: Tuple[bool, str, int]) -> Tuple[int, str, bool]

        Return a triple in reversed order
    """  # noqa: E501 line too long
    assert doc(ExamplePythonTypes.throw_exception) == """
        throw_exception(self: m.ExamplePythonTypes) -> None

        Throw an exception
    """
    assert doc(ExamplePythonTypes.new_instance) == """
        new_instance() -> m.ExamplePythonTypes

        Return an instance
    """


def test_module():
    import pybind11_tests

    assert pybind11_tests.__name__ == "pybind11_tests"
    assert ExamplePythonTypes.__name__ == "ExamplePythonTypes"
    assert ExamplePythonTypes.__module__ == "pybind11_tests"
    assert ExamplePythonTypes.get_set.__name__ == "get_set"
    assert ExamplePythonTypes.get_set.__module__ == "pybind11_tests"


def test_print(capture):
    from pybind11_tests import test_print_function

    with capture:
        test_print_function()
    assert capture == """
        Hello, World!
        1 2.0 three True -- multiple args
        *args-and-a-custom-separator
        no new line here -- next print
        flush
        py::print + str.format = this
    """
    assert capture.stderr == "this goes to stderr"


def test_str_api():
    from pybind11_tests import test_str_format

    s1, s2 = test_str_format()
    assert s1 == "1 + 2 = 3"
    assert s1 == s2


def test_dict_api():
    from pybind11_tests import test_dict_keyword_constructor

    assert test_dict_keyword_constructor() == {"x": 1, "y": 2, "z": 3}


def test_accessors():
    from pybind11_tests import test_accessor_api, test_tuple_accessor, test_accessor_assignment

    class SubTestObject:
        attr_obj = 1
        attr_char = 2

    class TestObject:
        basic_attr = 1
        begin_end = [1, 2, 3]
        d = {"operator[object]": 1, "operator[char *]": 2}
        sub = SubTestObject()

        def func(self, x, *args):
            return self.basic_attr + x + sum(args)

    d = test_accessor_api(TestObject())
    assert d["basic_attr"] == 1
    assert d["begin_end"] == [1, 2, 3]
    assert d["operator[object]"] == 1
    assert d["operator[char *]"] == 2
    assert d["attr(object)"] == 1
    assert d["attr(char *)"] == 2
    assert d["missing_attr_ptr"] == "raised"
    assert d["missing_attr_chain"] == "raised"
    assert d["is_none"] is False
    assert d["operator()"] == 2
    assert d["operator*"] == 7

    assert test_tuple_accessor(tuple()) == (0, 1, 2)

    d = test_accessor_assignment()
    assert d["get"] == 0
    assert d["deferred_get"] == 0
    assert d["set"] == 1
    assert d["deferred_set"] == 1
    assert d["var"] == 99


@pytest.mark.skipif(not has_optional, reason='no <optional>')
def test_optional():
    from pybind11_tests import double_or_zero, half_or_none, test_nullopt

    assert double_or_zero(None) == 0
    assert double_or_zero(42) == 84
    pytest.raises(TypeError, double_or_zero, 'foo')

    assert half_or_none(0) is None
    assert half_or_none(42) == 21
    pytest.raises(TypeError, half_or_none, 'foo')

    assert test_nullopt() == 42
    assert test_nullopt(None) == 42
    assert test_nullopt(42) == 42
    assert test_nullopt(43) == 43


@pytest.mark.skipif(not has_exp_optional, reason='no <experimental/optional>')
def test_exp_optional():
    from pybind11_tests import double_or_zero_exp, half_or_none_exp, test_nullopt_exp

    assert double_or_zero_exp(None) == 0
    assert double_or_zero_exp(42) == 84
    pytest.raises(TypeError, double_or_zero_exp, 'foo')

    assert half_or_none_exp(0) is None
    assert half_or_none_exp(42) == 21
    pytest.raises(TypeError, half_or_none_exp, 'foo')

    assert test_nullopt_exp() == 42
    assert test_nullopt_exp(None) == 42
    assert test_nullopt_exp(42) == 42
    assert test_nullopt_exp(43) == 43


def test_constructors():
    """C++ default and converting constructors are equivalent to type calls in Python"""
    from pybind11_tests import (test_default_constructors, test_converting_constructors,
                                test_cast_functions)

    types = [str, bool, int, float, tuple, list, dict, set]
    expected = {t.__name__: t() for t in types}
    assert test_default_constructors() == expected

    data = {
        str: 42,
        bool: "Not empty",
        int: "42",
        float: "+1e3",
        tuple: range(3),
        list: range(3),
        dict: [("two", 2), ("one", 1), ("three", 3)],
        set: [4, 4, 5, 6, 6, 6],
        memoryview: b'abc'
    }
    inputs = {k.__name__: v for k, v in data.items()}
    expected = {k.__name__: k(v) for k, v in data.items()}
    assert test_converting_constructors(inputs) == expected
    assert test_cast_functions(inputs) == expected


def test_move_out_container():
    """Properties use the `reference_internal` policy by default. If the underlying function
    returns an rvalue, the policy is automatically changed to `move` to avoid referencing
    a temporary. In case the return value is a container of user-defined types, the policy
    also needs to be applied to the elements, not just the container."""
    from pybind11_tests import MoveOutContainer

    c = MoveOutContainer()
    moved_out_list = c.move_list
    assert [x.value for x in moved_out_list] == [0, 1, 2]


def test_implicit_casting():
    """Tests implicit casting when assigning or appending to dicts and lists."""
    from pybind11_tests import get_implicit_casting

    z = get_implicit_casting()
    assert z['d'] == {
        'char*_i1': 'abc', 'char*_i2': 'abc', 'char*_e': 'abc', 'char*_p': 'abc',
        'str_i1': 'str', 'str_i2': 'str1', 'str_e': 'str2', 'str_p': 'str3',
        'int_i1': 42, 'int_i2': 42, 'int_e': 43, 'int_p': 44
    }
    assert z['l'] == [3, 6, 9, 12, 15]
