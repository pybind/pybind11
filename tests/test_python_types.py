import pytest

from pybind11_tests import ExamplePythonTypes, ConstructorStats


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
        set_result.add('key3')
        instance.print_set(set_result)
    assert capture.unordered == """
        key: key1
        key: key2
        key: key3
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
        Entry at positon 0: value
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
    array_result = instance.get_array()
    assert array_result == ['array entry 1', 'array entry 2']
    with capture:
        instance.print_array(array_result)
    assert capture.unordered == """
        array item 0: array entry 1
        array item 1: array entry 2
    """
    with pytest.raises(RuntimeError) as excinfo:
        instance.throw_exception()
    assert str(excinfo.value) == "This exception was intentionally thrown."

    assert instance.pair_passthrough((True, "test")) == ("test", True)
    assert instance.tuple_passthrough((True, "test", 5)) == (5, "test", True)

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


def test_docs(doc):
    assert doc(ExamplePythonTypes) == "Example 2 documentation"
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
    """
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
