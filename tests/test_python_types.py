# Python < 3 needs this: coding=utf-8
import pytest
import pybind11_tests

from pybind11_tests import ExamplePythonTypes, ConstructorStats, has_optional, has_exp_optional


def test_repr():
    # In Python 3.3+, repr() accesses __qualname__
    assert "pybind11_type" in repr(type(ExamplePythonTypes))
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
    from pybind11_tests import test_print_function, test_print_failure, debug_enabled

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

    with pytest.raises(RuntimeError) as excinfo:
        test_print_failure()
    assert str(excinfo.value) == "make_tuple(): unable to convert " + (
        "argument of type 'UnregisteredType' to Python object"
        if debug_enabled else
        "arguments to Python object (compile in debug mode for details)"
    )


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
    from pybind11_tests import (double_or_zero, half_or_none, test_nullopt,
                                test_no_assign, NoAssign)

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

    assert test_no_assign() == 42
    assert test_no_assign(None) == 42
    assert test_no_assign(NoAssign(43)) == 43
    pytest.raises(TypeError, test_no_assign, 43)


@pytest.mark.skipif(not has_exp_optional, reason='no <experimental/optional>')
def test_exp_optional():
    from pybind11_tests import (double_or_zero_exp, half_or_none_exp, test_nullopt_exp,
                                test_no_assign_exp, NoAssign)

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

    assert test_no_assign_exp() == 42
    assert test_no_assign_exp(None) == 42
    assert test_no_assign_exp(NoAssign(43)) == 43
    pytest.raises(TypeError, test_no_assign_exp, 43)


@pytest.mark.skipif(not hasattr(pybind11_tests, "load_variant"), reason='no <variant>')
def test_variant(doc):
    from pybind11_tests import load_variant, load_variant_2pass, cast_variant

    assert load_variant(1) == "int"
    assert load_variant("1") == "std::string"
    assert load_variant(1.0) == "double"
    assert load_variant(None) == "std::nullptr_t"

    assert load_variant_2pass(1) == "int"
    assert load_variant_2pass(1.0) == "double"

    assert cast_variant() == (5, "Hello")

    assert doc(load_variant) == "load_variant(arg0: Union[int, str, float, None]) -> str"


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


def test_unicode_conversion():
    """Tests unicode conversion and error reporting."""
    import pybind11_tests
    from pybind11_tests import (good_utf8_string, bad_utf8_string,
                                good_utf16_string, bad_utf16_string,
                                good_utf32_string,  # bad_utf32_string,
                                good_wchar_string,  # bad_wchar_string,
                                u8_Z, u8_eacute, u16_ibang, u32_mathbfA, wchar_heart)

    assert good_utf8_string() == u"Say utf8‽ 🎂 𝐀"
    assert good_utf16_string() == u"b‽🎂𝐀z"
    assert good_utf32_string() == u"a𝐀🎂‽z"
    assert good_wchar_string() == u"a⸘𝐀z"

    with pytest.raises(UnicodeDecodeError):
        bad_utf8_string()

    with pytest.raises(UnicodeDecodeError):
        bad_utf16_string()

    # These are provided only if they actually fail (they don't when 32-bit and under Python 2.7)
    if hasattr(pybind11_tests, "bad_utf32_string"):
        with pytest.raises(UnicodeDecodeError):
            pybind11_tests.bad_utf32_string()
    if hasattr(pybind11_tests, "bad_wchar_string"):
        with pytest.raises(UnicodeDecodeError):
            pybind11_tests.bad_wchar_string()

    assert u8_Z() == 'Z'
    assert u8_eacute() == u'é'
    assert u16_ibang() == u'‽'
    assert u32_mathbfA() == u'𝐀'
    assert wchar_heart() == u'♥'


def test_single_char_arguments():
    """Tests failures for passing invalid inputs to char-accepting functions"""
    from pybind11_tests import ord_char, ord_char16, ord_char32, ord_wchar, wchar_size

    def toobig_message(r):
        return "Character code point not in range({0:#x})".format(r)
    toolong_message = "Expected a character, but multi-character string found"

    assert ord_char(u'a') == 0x61  # simple ASCII
    assert ord_char(u'é') == 0xE9  # requires 2 bytes in utf-8, but can be stuffed in a char
    with pytest.raises(ValueError) as excinfo:
        assert ord_char(u'Ā') == 0x100  # requires 2 bytes, doesn't fit in a char
    assert str(excinfo.value) == toobig_message(0x100)
    with pytest.raises(ValueError) as excinfo:
        assert ord_char(u'ab')
    assert str(excinfo.value) == toolong_message

    assert ord_char16(u'a') == 0x61
    assert ord_char16(u'é') == 0xE9
    assert ord_char16(u'Ā') == 0x100
    assert ord_char16(u'‽') == 0x203d
    assert ord_char16(u'♥') == 0x2665
    with pytest.raises(ValueError) as excinfo:
        assert ord_char16(u'🎂') == 0x1F382  # requires surrogate pair
    assert str(excinfo.value) == toobig_message(0x10000)
    with pytest.raises(ValueError) as excinfo:
        assert ord_char16(u'aa')
    assert str(excinfo.value) == toolong_message

    assert ord_char32(u'a') == 0x61
    assert ord_char32(u'é') == 0xE9
    assert ord_char32(u'Ā') == 0x100
    assert ord_char32(u'‽') == 0x203d
    assert ord_char32(u'♥') == 0x2665
    assert ord_char32(u'🎂') == 0x1F382
    with pytest.raises(ValueError) as excinfo:
        assert ord_char32(u'aa')
    assert str(excinfo.value) == toolong_message

    assert ord_wchar(u'a') == 0x61
    assert ord_wchar(u'é') == 0xE9
    assert ord_wchar(u'Ā') == 0x100
    assert ord_wchar(u'‽') == 0x203d
    assert ord_wchar(u'♥') == 0x2665
    if wchar_size == 2:
        with pytest.raises(ValueError) as excinfo:
            assert ord_wchar(u'🎂') == 0x1F382  # requires surrogate pair
        assert str(excinfo.value) == toobig_message(0x10000)
    else:
        assert ord_wchar(u'🎂') == 0x1F382
    with pytest.raises(ValueError) as excinfo:
        assert ord_wchar(u'aa')
    assert str(excinfo.value) == toolong_message


def test_bytes_to_string():
    """Tests the ability to pass bytes to C++ string-accepting functions.  Note that this is
    one-way: the only way to return bytes to Python is via the pybind11::bytes class."""
    # Issue #816
    from pybind11_tests import strlen, string_length
    import sys
    byte = bytes if sys.version_info[0] < 3 else str

    assert strlen(byte("hi")) == 2
    assert string_length(byte("world")) == 5
    assert string_length(byte("a\x00b")) == 3
    assert strlen(byte("a\x00b")) == 1  # C-string limitation


def test_builtins_cast_return_none():
    """Casters produced with PYBIND11_TYPE_CASTER() should convert nullptr to None"""
    import pybind11_tests as m

    assert m.return_none_string() is None
    assert m.return_none_char() is None
    assert m.return_none_bool() is None
    assert m.return_none_int() is None
    assert m.return_none_float() is None


def test_none_deferred():
    """None passed as various argument types should defer to other overloads"""
    import pybind11_tests as m

    assert not m.defer_none_cstring("abc")
    assert m.defer_none_cstring(None)
    assert not m.defer_none_custom(m.ExamplePythonTypes.new_instance())
    assert m.defer_none_custom(None)
    assert m.nodefer_none_void(None)
    if has_optional:
        assert m.nodefer_none_optional(None)


def test_capsule_with_destructor(capture):
    import pybind11_tests as m
    pytest.gc_collect()
    with capture:
        a = m.return_capsule_with_destructor()
        del a
        pytest.gc_collect()
    assert capture.unordered == """
        creating capsule
        destructing capsule
    """

    with capture:
        a = m.return_capsule_with_destructor_2()
        del a
        pytest.gc_collect()
    assert capture.unordered == """
        creating capsule
        destructing capsule: 1234
    """


def test_void_caster():
    import pybind11_tests as m

    assert m.load_nullptr_t(None) is None
    assert m.cast_nullptr_t() is None


def test_reference_wrapper():
    """std::reference_wrapper<T> tests.

    #171: Can't return reference wrappers (or STL data structures containing them)
    #848: std::reference_wrapper accepts nullptr / None arguments [but shouldn't]
    (no issue): reference_wrappers should work for types with custom type casters
    """
    from pybind11_tests import (IntWrapper, return_vec_of_reference_wrapper, refwrap_int,
                                IncrIntWrapper, refwrap_iiw, refwrap_call_iiw,
                                refwrap_list_copies, refwrap_list_refs)

    # 171:
    assert str(return_vec_of_reference_wrapper(IntWrapper(4))) == \
        "[IntWrapper[1], IntWrapper[2], IntWrapper[3], IntWrapper[4]]"

    # 848:
    with pytest.raises(TypeError) as excinfo:
        return_vec_of_reference_wrapper(None)
    assert "incompatible function arguments" in str(excinfo.value)

    assert refwrap_int(42) == 420

    a1 = refwrap_list_copies()
    a2 = refwrap_list_copies()
    assert [x.i for x in a1] == [2, 3]
    assert [x.i for x in a2] == [2, 3]
    assert not a1[0] is a2[0] and not a1[1] is a2[1]

    b1 = refwrap_list_refs()
    b2 = refwrap_list_refs()
    assert [x.i for x in b1] == [1, 2]
    assert [x.i for x in b2] == [1, 2]
    assert b1[0] is b2[0] and b1[1] is b2[1]

    assert refwrap_iiw(IncrIntWrapper(5)) == 5
    assert refwrap_call_iiw(IncrIntWrapper(10), refwrap_iiw) == [10, 10, 10, 10]
