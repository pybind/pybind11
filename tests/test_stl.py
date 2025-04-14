from __future__ import annotations

import pytest

import env  # noqa: F401
from pybind11_tests import ConstructorStats, UserType
from pybind11_tests import stl as m


def test_vector(doc):
    """std::vector <-> list"""
    lst = m.cast_vector()
    assert lst == [1]
    lst.append(2)
    assert m.load_vector(lst)
    assert m.load_vector(tuple(lst))

    assert m.cast_bool_vector() == [True, False]
    assert m.load_bool_vector([True, False])
    assert m.load_bool_vector((True, False))

    assert doc(m.cast_vector) == "cast_vector() -> list[int]"
    assert (
        doc(m.load_vector)
        == "load_vector(arg0: collections.abc.Sequence[typing.SupportsInt]) -> bool"
    )

    # Test regression caused by 936: pointers to stl containers weren't castable
    assert m.cast_ptr_vector() == ["lvalue", "lvalue"]


def test_deque():
    """std::deque <-> list"""
    lst = m.cast_deque()
    assert lst == [1]
    lst.append(2)
    assert m.load_deque(lst)
    assert m.load_deque(tuple(lst))


def test_array(doc):
    """std::array <-> list"""
    lst = m.cast_array()
    assert lst == [1, 2]
    assert m.load_array(lst)
    assert m.load_array(tuple(lst))

    assert (
        doc(m.cast_array)
        == 'cast_array() -> typing.Annotated[list[int], "FixedSize(2)"]'
    )
    assert (
        doc(m.load_array)
        == 'load_array(arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"]) -> bool'
    )


def test_array_no_default_ctor():
    lst = m.NoDefaultCtorArray(3)
    assert [e.val for e in lst.arr] == [13, 23]
    lst.arr = m.NoDefaultCtorArray(4).arr
    assert [e.val for e in lst.arr] == [14, 24]


def test_valarray(doc):
    """std::valarray <-> list"""
    lst = m.cast_valarray()
    assert lst == [1, 4, 9]
    assert m.load_valarray(lst)
    assert m.load_valarray(tuple(lst))

    assert doc(m.cast_valarray) == "cast_valarray() -> list[int]"
    assert (
        doc(m.load_valarray)
        == "load_valarray(arg0: collections.abc.Sequence[typing.SupportsInt]) -> bool"
    )


def test_map(doc):
    """std::map <-> dict"""
    d = m.cast_map()
    assert d == {"key": "value"}
    assert "key" in d
    d["key2"] = "value2"
    assert "key2" in d
    assert m.load_map(d)

    assert doc(m.cast_map) == "cast_map() -> dict[str, str]"
    assert (
        doc(m.load_map) == "load_map(arg0: collections.abc.Mapping[str, str]) -> bool"
    )


def test_set(doc):
    """std::set <-> set"""
    s = m.cast_set()
    assert s == {"key1", "key2"}
    s.add("key3")
    assert m.load_set(s)
    assert m.load_set(frozenset(s))

    assert doc(m.cast_set) == "cast_set() -> set[str]"
    assert doc(m.load_set) == "load_set(arg0: collections.abc.Set[str]) -> bool"


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
        "b": [[["lvalue", "lvalue"], ["lvalue", "lvalue"]]],
    }

    # Issue #853 test case:
    z = m.cast_unique_ptr_vector()
    assert z[0].value == 7
    assert z[1].value == 42


def test_move_out_container():
    """Properties use the `reference_internal` policy by default. If the underlying function
    returns an rvalue, the policy is automatically changed to `move` to avoid referencing
    a temporary. In case the return value is a container of user-defined types, the policy
    also needs to be applied to the elements, not just the container."""
    c = m.MoveOutContainer()
    moved_out_list = c.move_list
    assert [x.value for x in moved_out_list] == [0, 1, 2]


@pytest.mark.skipif(not hasattr(m, "has_optional"), reason="no <optional>")
def test_optional():
    assert m.double_or_zero(None) == 0
    assert m.double_or_zero(42) == 84
    pytest.raises(TypeError, m.double_or_zero, "foo")

    assert m.half_or_none(0) is None
    assert m.half_or_none(42) == 21
    pytest.raises(TypeError, m.half_or_none, "foo")

    assert m.test_nullopt() == 42
    assert m.test_nullopt(None) == 42
    assert m.test_nullopt(42) == 42
    assert m.test_nullopt(43) == 43

    assert m.test_no_assign() == 42
    assert m.test_no_assign(None) == 42
    assert m.test_no_assign(m.NoAssign(43)) == 43
    pytest.raises(TypeError, m.test_no_assign, 43)

    assert m.nodefer_none_optional(None)

    holder = m.OptionalHolder()
    mvalue = holder.member
    assert mvalue.initialized
    assert holder.member_initialized()

    props = m.OptionalProperties()
    assert int(props.access_by_ref) == 42
    assert int(props.access_by_copy) == 42


@pytest.mark.skipif(
    not hasattr(m, "has_exp_optional"), reason="no <experimental/optional>"
)
def test_exp_optional():
    assert m.double_or_zero_exp(None) == 0
    assert m.double_or_zero_exp(42) == 84
    pytest.raises(TypeError, m.double_or_zero_exp, "foo")

    assert m.half_or_none_exp(0) is None
    assert m.half_or_none_exp(42) == 21
    pytest.raises(TypeError, m.half_or_none_exp, "foo")

    assert m.test_nullopt_exp() == 42
    assert m.test_nullopt_exp(None) == 42
    assert m.test_nullopt_exp(42) == 42
    assert m.test_nullopt_exp(43) == 43

    assert m.test_no_assign_exp() == 42
    assert m.test_no_assign_exp(None) == 42
    assert m.test_no_assign_exp(m.NoAssign(43)) == 43
    pytest.raises(TypeError, m.test_no_assign_exp, 43)

    holder = m.OptionalExpHolder()
    mvalue = holder.member
    assert mvalue.initialized
    assert holder.member_initialized()

    props = m.OptionalExpProperties()
    assert int(props.access_by_ref) == 42
    assert int(props.access_by_copy) == 42


@pytest.mark.skipif(not hasattr(m, "has_boost_optional"), reason="no <boost/optional>")
def test_boost_optional():
    assert m.double_or_zero_boost(None) == 0
    assert m.double_or_zero_boost(42) == 84
    pytest.raises(TypeError, m.double_or_zero_boost, "foo")

    assert m.half_or_none_boost(0) is None
    assert m.half_or_none_boost(42) == 21
    pytest.raises(TypeError, m.half_or_none_boost, "foo")

    assert m.test_nullopt_boost() == 42
    assert m.test_nullopt_boost(None) == 42
    assert m.test_nullopt_boost(42) == 42
    assert m.test_nullopt_boost(43) == 43

    assert m.test_no_assign_boost() == 42
    assert m.test_no_assign_boost(None) == 42
    assert m.test_no_assign_boost(m.NoAssign(43)) == 43
    pytest.raises(TypeError, m.test_no_assign_boost, 43)

    holder = m.OptionalBoostHolder()
    mvalue = holder.member
    assert mvalue.initialized
    assert holder.member_initialized()

    props = m.OptionalBoostProperties()
    assert int(props.access_by_ref) == 42
    assert int(props.access_by_copy) == 42


def test_reference_sensitive_optional():
    assert m.double_or_zero_refsensitive(None) == 0
    assert m.double_or_zero_refsensitive(42) == 84
    pytest.raises(TypeError, m.double_or_zero_refsensitive, "foo")

    assert m.half_or_none_refsensitive(0) is None
    assert m.half_or_none_refsensitive(42) == 21
    pytest.raises(TypeError, m.half_or_none_refsensitive, "foo")

    assert m.test_nullopt_refsensitive() == 42
    assert m.test_nullopt_refsensitive(None) == 42
    assert m.test_nullopt_refsensitive(42) == 42
    assert m.test_nullopt_refsensitive(43) == 43

    assert m.test_no_assign_refsensitive() == 42
    assert m.test_no_assign_refsensitive(None) == 42
    assert m.test_no_assign_refsensitive(m.NoAssign(43)) == 43
    pytest.raises(TypeError, m.test_no_assign_refsensitive, 43)

    holder = m.OptionalRefSensitiveHolder()
    mvalue = holder.member
    assert mvalue.initialized
    assert holder.member_initialized()

    props = m.OptionalRefSensitiveProperties()
    assert int(props.access_by_ref) == 42
    assert int(props.access_by_copy) == 42


@pytest.mark.skipif(not hasattr(m, "has_filesystem"), reason="no <filesystem>")
def test_fs_path(doc):
    from pathlib import Path

    class PseudoStrPath:
        def __fspath__(self):
            return "foo/bar"

    class PseudoBytesPath:
        def __fspath__(self):
            return b"foo/bar"

    # Single argument
    assert m.parent_path(Path("foo/bar")) == Path("foo")
    assert m.parent_path("foo/bar") == Path("foo")
    assert m.parent_path(b"foo/bar") == Path("foo")
    assert m.parent_path(PseudoStrPath()) == Path("foo")
    assert m.parent_path(PseudoBytesPath()) == Path("foo")
    assert (
        doc(m.parent_path)
        == "parent_path(arg0: Union[os.PathLike, str, bytes]) -> pathlib.Path"
    )
    # std::vector
    assert m.parent_paths(["foo/bar", "foo/baz"]) == [Path("foo"), Path("foo")]
    assert (
        doc(m.parent_paths)
        == "parent_paths(arg0: collections.abc.Sequence[Union[os.PathLike, str, bytes]]) -> list[pathlib.Path]"
    )
    # py::typing::List
    assert m.parent_paths_list(["foo/bar", "foo/baz"]) == [Path("foo"), Path("foo")]
    assert (
        doc(m.parent_paths_list)
        == "parent_paths_list(arg0: list[Union[os.PathLike, str, bytes]]) -> list[pathlib.Path]"
    )
    # Nested py::typing::List
    assert m.parent_paths_nested_list([["foo/bar"], ["foo/baz", "foo/buzz"]]) == [
        [Path("foo")],
        [Path("foo"), Path("foo")],
    ]
    assert (
        doc(m.parent_paths_nested_list)
        == "parent_paths_nested_list(arg0: list[list[Union[os.PathLike, str, bytes]]]) -> list[list[pathlib.Path]]"
    )
    # py::typing::Tuple
    assert m.parent_paths_tuple(("foo/bar", "foo/baz")) == (Path("foo"), Path("foo"))
    assert (
        doc(m.parent_paths_tuple)
        == "parent_paths_tuple(arg0: tuple[Union[os.PathLike, str, bytes], Union[os.PathLike, str, bytes]]) -> tuple[pathlib.Path, pathlib.Path]"
    )
    # py::typing::Dict
    assert m.parent_paths_dict(
        {
            "key1": Path("foo/bar"),
            "key2": "foo/baz",
            "key3": b"foo/buzz",
        }
    ) == {
        "key1": Path("foo"),
        "key2": Path("foo"),
        "key3": Path("foo"),
    }
    assert (
        doc(m.parent_paths_dict)
        == "parent_paths_dict(arg0: dict[str, Union[os.PathLike, str, bytes]]) -> dict[str, pathlib.Path]"
    )


@pytest.mark.skipif(not hasattr(m, "load_variant"), reason="no <variant>")
def test_variant(doc):
    assert m.load_variant(1) == "int"
    assert m.load_variant("1") == "std::string"
    assert m.load_variant(1.0) == "double"
    assert m.load_variant(None) == "std::nullptr_t"

    assert m.load_variant_2pass(1) == "int"
    assert m.load_variant_2pass(1.0) == "double"

    assert m.cast_variant() == (5, "Hello")

    assert (
        doc(m.load_variant)
        == "load_variant(arg0: Union[typing.SupportsInt, str, typing.SupportsFloat, None]) -> str"
    )


@pytest.mark.skipif(
    not hasattr(m, "load_monostate_variant"), reason="no std::monostate"
)
def test_variant_monostate(doc):
    assert m.load_monostate_variant(None) == "std::monostate"
    assert m.load_monostate_variant(1) == "int"
    assert m.load_monostate_variant("1") == "std::string"

    assert m.cast_monostate_variant() == (None, 5, "Hello")

    assert (
        doc(m.load_monostate_variant)
        == "load_monostate_variant(arg0: Union[None, typing.SupportsInt, str]) -> str"
    )


def test_vec_of_reference_wrapper():
    """#171: Can't return reference wrappers (or STL structures containing them)"""
    assert (
        str(m.return_vec_of_reference_wrapper(UserType(4)))
        == "[UserType(1), UserType(2), UserType(3), UserType(4)]"
    )


def test_stl_pass_by_pointer(msg):
    """Passing nullptr or None to an STL container pointer is not expected to work"""
    with pytest.raises(TypeError) as excinfo:
        m.stl_pass_by_pointer()  # default value is `nullptr`
    assert (
        msg(excinfo.value)
        == """
        stl_pass_by_pointer(): incompatible function arguments. The following argument types are supported:
            1. (v: collections.abc.Sequence[typing.SupportsInt] = None) -> list[int]

        Invoked with:
    """
    )

    with pytest.raises(TypeError) as excinfo:
        m.stl_pass_by_pointer(None)
    assert (
        msg(excinfo.value)
        == """
        stl_pass_by_pointer(): incompatible function arguments. The following argument types are supported:
            1. (v: collections.abc.Sequence[typing.SupportsInt] = None) -> list[int]

        Invoked with: None
    """
    )

    assert m.stl_pass_by_pointer([1, 2, 3]) == [1, 2, 3]


def test_missing_header_message():
    """Trying convert `list` to a `std::vector`, or vice versa, without including
    <pybind11/stl.h> should result in a helpful suggestion in the error message"""
    import pybind11_cross_module_tests as cm

    expected_message = (
        "Did you forget to `#include <pybind11/stl.h>`? Or <pybind11/complex.h>,\n"
        "<pybind11/functional.h>, <pybind11/chrono.h>, etc. Some automatic\n"
        "conversions are optional and require extra headers to be included\n"
        "when compiling your pybind11 module."
    )

    with pytest.raises(TypeError) as excinfo:
        cm.missing_header_arg([1.0, 2.0, 3.0])
    assert expected_message in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        cm.missing_header_return()
    assert expected_message in str(excinfo.value)


def test_function_with_string_and_vector_string_arg():
    """Check if a string is NOT implicitly converted to a list, which was the
    behavior before fix of issue #1258"""
    assert m.func_with_string_or_vector_string_arg_overload(("A", "B")) == 2
    assert m.func_with_string_or_vector_string_arg_overload(["A", "B"]) == 2
    assert m.func_with_string_or_vector_string_arg_overload("A") == 3


@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
def test_stl_ownership():
    cstats = ConstructorStats.get(m.Placeholder)
    assert cstats.alive() == 0
    r = m.test_stl_ownership()
    assert len(r) == 1
    del r
    assert cstats.alive() == 0


def test_array_cast_sequence():
    assert m.array_cast_sequence((1, 2, 3)) == [1, 2, 3]


def test_issue_1561():
    """check fix for issue #1561"""
    bar = m.Issue1561Outer()
    bar.list = [m.Issue1561Inner("bar")]
    assert bar.list
    assert bar.list[0].data == "bar"


def test_return_vector_bool_raw_ptr():
    # Add `while True:` for manual leak checking.
    v = m.return_vector_bool_raw_ptr()
    assert isinstance(v, list)
    assert len(v) == 4513


@pytest.mark.parametrize(
    ("fn", "offset"), [(m.pass_std_vector_int, 0), (m.pass_std_array_int_2, 1)]
)
def test_pass_std_vector_int(fn, offset):
    assert fn([7, 13]) == 140 + offset
    assert fn({6, 2}) == 116 + offset
    assert fn({"x": 8, "y": 11}.values()) == 138 + offset
    assert fn({3: None, 9: None}.keys()) == 124 + offset
    assert fn(i for i in [4, 17]) == 142 + offset
    assert fn(map(lambda i: i * 3, [8, 7])) == 190 + offset  # noqa: C417
    with pytest.raises(TypeError):
        fn({"x": 0, "y": 1})
    with pytest.raises(TypeError):
        fn({})


def test_pass_std_vector_pair_int():
    fn = m.pass_std_vector_pair_int
    assert fn({1: 2, 3: 4}.items()) == 406
    assert fn(zip([5, 17], [13, 9])) == 2222


def test_list_caster_fully_consumes_generator_object():
    def gen_invalid():
        yield from [1, 2.0, 3]

    gen_obj = gen_invalid()
    with pytest.raises(TypeError):
        m.pass_std_vector_int(gen_obj)
    assert not tuple(gen_obj)


def test_pass_std_set_int():
    fn = m.pass_std_set_int
    assert fn({3, 15}) == 254
    assert fn({5: None, 12: None}.keys()) == 251
    with pytest.raises(TypeError):
        fn([])
    with pytest.raises(TypeError):
        fn({})
    with pytest.raises(TypeError):
        fn({}.values())
    with pytest.raises(TypeError):
        fn(i for i in [])


def test_set_caster_dict_keys_failure():
    dict_keys = {1: None, 2.0: None, 3: None}.keys()
    # The asserts does not really exercise anything in pybind11, but if one of
    # them fails in some future version of Python, the set_caster load
    # implementation may need to be revisited.
    assert tuple(dict_keys) == (1, 2.0, 3)
    assert tuple(dict_keys) == (1, 2.0, 3)
    with pytest.raises(TypeError):
        m.pass_std_set_int(dict_keys)
    assert tuple(dict_keys) == (1, 2.0, 3)


class FakePyMappingMissingItems:
    def __getitem__(self, _):
        raise RuntimeError("Not expected to be called.")


class FakePyMappingWithItems(FakePyMappingMissingItems):
    def items(self):
        return ((1, 3), (2, 4))


class FakePyMappingBadItems(FakePyMappingMissingItems):
    def items(self):
        return ((1, 2), (3, "x"))


class FakePyMappingItemsNotCallable(FakePyMappingMissingItems):
    @property
    def items(self):
        return ((1, 2), (3, 4))


class FakePyMappingItemsWithArg(FakePyMappingMissingItems):
    def items(self, _):
        return ((1, 2), (3, 4))


class FakePyMappingGenObj(FakePyMappingMissingItems):
    def __init__(self, gen_obj):
        super().__init__()
        self.gen_obj = gen_obj

    def items(self):
        yield from self.gen_obj


def test_pass_std_map_int():
    fn = m.pass_std_map_int
    assert fn({1: 2, 3: 4}) == 4506
    with pytest.raises(TypeError):
        fn([])
    assert fn(FakePyMappingWithItems()) == 3507
    with pytest.raises(TypeError):
        fn(FakePyMappingMissingItems())
    with pytest.raises(TypeError):
        fn(FakePyMappingBadItems())
    with pytest.raises(TypeError):
        fn(FakePyMappingItemsNotCallable())
    with pytest.raises(TypeError):
        fn(FakePyMappingItemsWithArg())


@pytest.mark.parametrize(
    ("items", "expected_exception"),
    [
        (((1, 2), (3, "x"), (4, 5)), TypeError),
        (((1, 2), (3, 4, 5), (6, 7)), ValueError),
    ],
)
def test_map_caster_fully_consumes_generator_object(items, expected_exception):
    def gen_invalid():
        yield from items

    gen_obj = gen_invalid()
    with pytest.raises(expected_exception):
        m.pass_std_map_int(FakePyMappingGenObj(gen_obj))
    assert not tuple(gen_obj)


def test_sequence_caster_protocol(doc):
    from collections.abc import Sequence

    # Implements the Sequence protocol without explicitly inheriting from collections.abc.Sequence.
    class BareSequenceLike:
        def __init__(self, *args):
            self.data = tuple(args)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    # Implements the Sequence protocol by reusing BareSequenceLike's implementation.
    # Additionally, inherits from collections.abc.Sequence.
    class FormalSequenceLike(BareSequenceLike, Sequence):
        pass

    # convert mode
    assert (
        doc(m.roundtrip_std_vector_int)
        == "roundtrip_std_vector_int(arg0: collections.abc.Sequence[typing.SupportsInt]) -> list[int]"
    )
    assert m.roundtrip_std_vector_int([1, 2, 3]) == [1, 2, 3]
    assert m.roundtrip_std_vector_int((1, 2, 3)) == [1, 2, 3]
    assert m.roundtrip_std_vector_int(FormalSequenceLike(1, 2, 3)) == [1, 2, 3]
    assert m.roundtrip_std_vector_int(BareSequenceLike(1, 2, 3)) == [1, 2, 3]
    assert m.roundtrip_std_vector_int([]) == []
    assert m.roundtrip_std_vector_int(()) == []
    assert m.roundtrip_std_vector_int(BareSequenceLike()) == []
    # noconvert mode
    assert (
        doc(m.roundtrip_std_vector_int_noconvert)
        == "roundtrip_std_vector_int_noconvert(v: list[int]) -> list[int]"
    )
    assert m.roundtrip_std_vector_int_noconvert([1, 2, 3]) == [1, 2, 3]
    assert m.roundtrip_std_vector_int_noconvert((1, 2, 3)) == [1, 2, 3]
    assert m.roundtrip_std_vector_int_noconvert(FormalSequenceLike(1, 2, 3)) == [
        1,
        2,
        3,
    ]
    assert m.roundtrip_std_vector_int_noconvert(BareSequenceLike(1, 2, 3)) == [1, 2, 3]
    assert m.roundtrip_std_vector_int_noconvert([]) == []
    assert m.roundtrip_std_vector_int_noconvert(()) == []
    assert m.roundtrip_std_vector_int_noconvert(BareSequenceLike()) == []


def test_mapping_caster_protocol(doc):
    from collections.abc import Mapping

    # Implements the Mapping protocol without explicitly inheriting from collections.abc.Mapping.
    class BareMappingLike:
        def __init__(self, **kwargs):
            self.data = dict(kwargs)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, key):
            return self.data[key]

        def __iter__(self):
            yield from self.data

    # Implements the Mapping protocol by reusing BareMappingLike's implementation.
    # Additionally, inherits from collections.abc.Mapping.
    class FormalMappingLike(BareMappingLike, Mapping):
        pass

    a1b2c3 = {"a": 1, "b": 2, "c": 3}
    # convert mode
    assert (
        doc(m.roundtrip_std_map_str_int)
        == "roundtrip_std_map_str_int(arg0: collections.abc.Mapping[str, typing.SupportsInt]) -> dict[str, int]"
    )
    assert m.roundtrip_std_map_str_int(a1b2c3) == a1b2c3
    assert m.roundtrip_std_map_str_int(FormalMappingLike(**a1b2c3)) == a1b2c3
    assert m.roundtrip_std_map_str_int({}) == {}
    assert m.roundtrip_std_map_str_int(FormalMappingLike()) == {}
    with pytest.raises(TypeError):
        m.roundtrip_std_map_str_int(BareMappingLike(**a1b2c3))
    # noconvert mode
    assert (
        doc(m.roundtrip_std_map_str_int_noconvert)
        == "roundtrip_std_map_str_int_noconvert(m: dict[str, int]) -> dict[str, int]"
    )
    assert m.roundtrip_std_map_str_int_noconvert(a1b2c3) == a1b2c3
    assert m.roundtrip_std_map_str_int_noconvert({}) == {}
    with pytest.raises(TypeError):
        m.roundtrip_std_map_str_int_noconvert(FormalMappingLike(**a1b2c3))
    with pytest.raises(TypeError):
        m.roundtrip_std_map_str_int_noconvert(BareMappingLike(**a1b2c3))


def test_set_caster_protocol(doc):
    from collections.abc import Set

    # Implements the Set protocol without explicitly inheriting from collections.abc.Set.
    class BareSetLike:
        def __init__(self, *args):
            self.data = set(args)

        def __len__(self):
            return len(self.data)

        def __contains__(self, item):
            return item in self.data

        def __iter__(self):
            yield from self.data

    # Implements the Set protocol by reusing BareSetLike's implementation.
    # Additionally, inherits from collections.abc.Set.
    class FormalSetLike(BareSetLike, Set):
        pass

    # convert mode
    assert (
        doc(m.roundtrip_std_set_int)
        == "roundtrip_std_set_int(arg0: collections.abc.Set[typing.SupportsInt]) -> set[int]"
    )
    assert m.roundtrip_std_set_int({1, 2, 3}) == {1, 2, 3}
    assert m.roundtrip_std_set_int(FormalSetLike(1, 2, 3)) == {1, 2, 3}
    assert m.roundtrip_std_set_int(set()) == set()
    assert m.roundtrip_std_set_int(FormalSetLike()) == set()
    with pytest.raises(TypeError):
        m.roundtrip_std_set_int(BareSetLike(1, 2, 3))
    # noconvert mode
    assert (
        doc(m.roundtrip_std_set_int_noconvert)
        == "roundtrip_std_set_int_noconvert(s: set[int]) -> set[int]"
    )
    assert m.roundtrip_std_set_int_noconvert({1, 2, 3}) == {1, 2, 3}
    assert m.roundtrip_std_set_int_noconvert(set()) == set()
    with pytest.raises(TypeError):
        m.roundtrip_std_set_int_noconvert(FormalSetLike(1, 2, 3))
    with pytest.raises(TypeError):
        m.roundtrip_std_set_int_noconvert(BareSetLike(1, 2, 3))
