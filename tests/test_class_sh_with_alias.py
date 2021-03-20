# -*- coding: utf-8 -*-
import re
import pytest
import env  # noqa: F401

from pybind11_tests import class_sh_with_alias as m


def check_regex(expected, actual):
    result = re.match(expected + "$", actual)
    if result is None:
        pytest.fail("expected: '{}' != actual: '{}'".format(expected, actual))


class PyDrvd0(m.Abase0):
    def __init__(self, val):
        super(PyDrvd0, self).__init__(val)

    def Add(self, other_val):  # noqa:  N802
        return self.Get() * 100 + other_val


class PyDrvd1(m.Abase1):
    def __init__(self, val):
        super(PyDrvd1, self).__init__(val)

    def Add(self, other_val):  # noqa:  N802
        return self.Get() * 200 + other_val


def test_drvd0_add():
    drvd = PyDrvd0(74)
    assert drvd.Add(38) == (74 * 10 + 3) * 100 + 38


def test_drvd0_add_in_cpp_raw_ptr():
    drvd = PyDrvd0(52)
    assert m.AddInCppRawPtr(drvd, 27) == ((52 * 10 + 3) * 100 + 27) * 10 + 7


def test_drvd0_add_in_cpp_shared_ptr():
    while True:
        drvd = PyDrvd0(36)
        assert m.AddInCppSharedPtr(drvd, 56) == ((36 * 10 + 3) * 100 + 56) * 100 + 11
        return  # Comment out for manual leak checking (use `top` command).


def test_drvd0_add_in_cpp_unique_ptr():
    while True:
        drvd = PyDrvd0(0)
        with pytest.raises(ValueError) as exc_info:
            m.AddInCppUniquePtr(drvd, 0)
        assert (
            str(exc_info.value)
            == "Alias class (also known as trampoline) does not inherit from"
            " py::virtual_overrider_self_life_support, therefore the ownership of this"
            " instance cannot safely be transferred to C++."
        )
        return  # Comment out for manual leak checking (use `top` command).


def test_drvd1_add_in_cpp_unique_ptr():
    while True:
        drvd = PyDrvd1(25)
        assert m.AddInCppUniquePtr(drvd, 83) == ((25 * 10 + 3) * 200 + 83) * 100 + 13
        return  # Comment out for manual leak checking (use `top` command).


class PyConsumer1(m.ConsumerBase):
    def __init__(self):
        m.ConsumerBase.__init__(self)

    def pass_uq_cref(self, obj):
        obj.mtxt = obj.mtxt + "pass_uq_cref"

    def pass_valu(self, obj):
        obj.mtxt = obj.mtxt + "pass_valu"

    def pass_lref(self, obj):
        obj.mtxt = obj.mtxt + "pass_lref"

    def pass_cref(self, obj):
        obj.mtxt = obj.mtxt + "pass_cref"


class PyConsumer2(m.ConsumerBase):
    """This one, additionally to PyConsumer1 calls the base methods.
    This results in a second call to the trampoline override dispatcher.
    Hence arguments have travelled a long way back and forth between C++
    and Python: C++ -> Python (call #1) -> C++ (call #2)."""

    def __init__(self):
        m.ConsumerBase.__init__(self)

    def pass_uq_cref(self, obj):
        obj.mtxt = obj.mtxt + "pass_uq_cref"
        m.ConsumerBase.pass_uq_cref(self, obj)

    def pass_valu(self, obj):
        obj.mtxt = obj.mtxt + "pass_valu"
        m.ConsumerBase.pass_valu(self, obj)

    def pass_lref(self, obj):
        obj.mtxt = obj.mtxt + "pass_lref"
        m.ConsumerBase.pass_lref(self, obj)

    def pass_cref(self, obj):
        obj.mtxt = obj.mtxt + "pass_cref"
        m.ConsumerBase.pass_cref(self, obj)


# roundtrip tests, creating an object in C++ that is passed by reference
# to a virtual method of a class derived in Python. Thus:
# C++ -> Python -> C++
@pytest.mark.parametrize(
    "f, expected",
    [
        (m.check_roundtrip_uq_cref, "([0-9]+)_pass_uq_cref"),
        (m.check_roundtrip_valu, "([0-9]+)_"),  # modification not passed back to C++
        (m.check_roundtrip_lref, "([0-9]+)_pass_lref"),
        pytest.param(
            m.check_roundtrip_cref,
            "([0-9]+)_pass_cref",
            marks=pytest.mark.skipif("env.PYPY"),
        ),
    ],
)
def test_unique_ptr_consumer1_roundtrip(f, expected):
    c = PyConsumer1()
    check_regex(expected, f(c))


@pytest.mark.parametrize(
    "f, expected",
    [
        pytest.param(  # cannot (yet) load unowned const unique_ptr& (for 2nd call)
            m.check_roundtrip_uq_cref,
            "([0-9]+)_pass_uq_cref_\\1",
            marks=pytest.mark.xfail,
        ),
        (m.check_roundtrip_valu, "([0-9]+)_"),  # modification not passed back to C++
        (m.check_roundtrip_lref, "([0-9]+)_pass_lref_\\1"),
        pytest.param(  # PYPY always copies the argument instead of passing the reference
            m.check_roundtrip_cref,
            "([0-9]+)_pass_cref_\\1",
            marks=pytest.mark.skipif("env.PYPY"),
        ),
    ],
)
def test_unique_ptr_consumer2_roundtrip(f, expected):
    c = PyConsumer2()
    check_regex(expected, f(c))
