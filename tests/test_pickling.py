from __future__ import annotations

import pickle
import re

import pytest

import env
from pybind11_tests import pickling as m


def all_pickle_protocols():
    assert pickle.HIGHEST_PROTOCOL >= 0
    return range(pickle.HIGHEST_PROTOCOL + 1)


@pytest.mark.parametrize("protocol", all_pickle_protocols())
def test_pickle_simple_callable(protocol):
    assert m.simple_callable() == 20220426
    serialized = pickle.dumps(m.simple_callable, protocol=protocol)
    assert b"pybind11_tests.pickling" in serialized
    assert b"simple_callable" in serialized
    deserialized = pickle.loads(serialized)
    assert deserialized() == 20220426
    assert deserialized is m.simple_callable

    # UNUSUAL: function record pickle roundtrip returns a module, not a function record object:
    if not env.PYPY:
        assert (
            pickle.loads(pickle.dumps(m.simple_callable.__self__, protocol=protocol))
            is m
        )
    # This is not expected to create issues because the only purpose of
    # `m.simple_callable.__self__` is to enable pickling: the only method it has is
    # `__reduce_ex__`. Direct access for any other purpose is not supported.
    # Note that `repr(m.simple_callable.__self__)` shows, e.g.:
    # `<pybind11_detail_function_record_v1__gcc_libstdcpp_cxxabi1018 object at 0x...>`
    # It is considered to be as much an implementation detail as the
    # `pybind11::detail::function_record` C++ type is.

    # @rainwoodman suggested that the unusual pickle roundtrip behavior could be
    # avoided by changing `reduce_ex_impl()` to produce, e.g.:
    # `"__import__('importlib').import_module('pybind11_tests.pickling').simple_callable.__self__"`
    # as the argument for the `eval()` function, and adding a getter to the
    # `function_record_PyTypeObject` that returns `self`. However, the additional code complexity
    # for this is deemed warranted only if the unusual pickle roundtrip behavior actually
    # creates issues.


@pytest.mark.parametrize("cls_name", ["Pickleable", "PickleableNew"])
def test_roundtrip(cls_name):
    cls = getattr(m, cls_name)
    p = cls("test_value")
    p.setExtra1(15)
    p.setExtra2(48)

    data = pickle.dumps(p, 2)  # Must use pickle protocol >= 2
    p2 = pickle.loads(data)
    assert p2.value() == p.value()
    assert p2.extra1() == p.extra1()
    assert p2.extra2() == p.extra2()


@pytest.mark.xfail("env.PYPY")
@pytest.mark.parametrize("cls_name", ["PickleableWithDict", "PickleableWithDictNew"])
def test_roundtrip_with_dict(cls_name):
    cls = getattr(m, cls_name)
    p = cls("test_value")
    p.extra = 15
    p.dynamic = "Attribute"

    data = pickle.dumps(p, pickle.HIGHEST_PROTOCOL)
    p2 = pickle.loads(data)
    assert p2.value == p.value
    assert p2.extra == p.extra
    assert p2.dynamic == p.dynamic


def test_enum_pickle():
    from pybind11_tests import enums as e

    data = pickle.dumps(e.EOne, 2)
    assert e.EOne == pickle.loads(data)


#
# exercise_trampoline
#
class SimplePyDerived(m.SimpleBase):
    pass


def test_roundtrip_simple_py_derived():
    p = SimplePyDerived()
    p.num = 202
    p.stored_in_dict = 303
    data = pickle.dumps(p, pickle.HIGHEST_PROTOCOL)
    p2 = pickle.loads(data)
    assert isinstance(p2, SimplePyDerived)
    assert p2.num == 202
    assert p2.stored_in_dict == 303


def test_roundtrip_simple_cpp_derived():
    p = m.make_SimpleCppDerivedAsBase()
    assert m.check_dynamic_cast_SimpleCppDerived(p)
    p.num = 404
    if not env.PYPY:
        # To ensure that this unit test is not accidentally invalidated.
        with pytest.raises(AttributeError):
            # Mimics the `setstate` C++ implementation.
            setattr(p, "__dict__", {})  # noqa: B010
    data = pickle.dumps(p, pickle.HIGHEST_PROTOCOL)
    p2 = pickle.loads(data)
    assert isinstance(p2, m.SimpleBase)
    assert p2.num == 404
    # Issue #3062: pickleable base C++ classes can incur object slicing
    #              if derived typeid is not registered with pybind11
    assert not m.check_dynamic_cast_SimpleCppDerived(p2)


def test_new_style_pickle_getstate_pos_only():
    assert (
        re.match(
            r"^__getstate__\(self: [\w\.]+, /\)", m.PickleableNew.__getstate__.__doc__
        )
        is not None
    )
    if hasattr(m, "PickleableWithDictNew"):
        assert (
            re.match(
                r"^__getstate__\(self: [\w\.]+, /\)",
                m.PickleableWithDictNew.__getstate__.__doc__,
            )
            is not None
        )
