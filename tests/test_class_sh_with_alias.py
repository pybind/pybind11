# -*- coding: utf-8 -*-
import pytest
import env  # noqa: F401

from pybind11_tests import class_sh_with_alias as m


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


# Python class inheriting from C++ class ReferencePassingTest
# virtual methods modify the obj's mtxt, which should become visible in C++
# To ensure that the original object instance was passed through,
# the pointer id of the received obj is returned by all pass_*() functions
# (and compared by the C++ caller with the originally passed obj id).
class PyReferencePassingTest1(m.ReferencePassingTest):
    def __init__(self):
        m.ReferencePassingTest.__init__(self)

    def pass_uq_cref(self, obj):
        obj.mtxt = obj.mtxt + "pass_uq_cref"
        return obj.id

    def pass_valu(self, obj):
        obj.mtxt = obj.mtxt + "pass_valu"
        return obj.id

    def pass_mref(self, obj):
        obj.mtxt = obj.mtxt + "pass_mref"
        return obj.id

    def pass_mptr(self, obj):
        obj.mtxt = obj.mtxt + "pass_mptr"
        return obj.id

    def pass_cref(self, obj):
        with pytest.raises(Exception):  # should be forbidden
            obj.mtxt = obj.mtxt + "pass_cref"
        return obj.id

    def pass_cptr(self, obj):
        with pytest.raises(Exception):  # should be forbidden
            obj.mtxt = obj.mtxt + "pass_cptr"
        return obj.id


# This class, in contrast to PyReferencePassingTest1, calls the base class methods as well,
# which will augment mtxt with a _MODIFIED stamp.
# These calls to the base class methods actually result in a 2nd call to the
# trampoline override dispatcher, requiring argument loading, which should pass
# references through as well, to make these tests succeed.
# argument is passed like this: C++ -> Python (call #1) -> C++ (call #2).
class PyReferencePassingTest2(m.ReferencePassingTest):
    def __init__(self):
        m.ReferencePassingTest.__init__(self)

    def pass_uq_cref(self, obj):
        obj.mtxt = obj.mtxt + "pass_uq_cref"
        return m.ReferencePassingTest.pass_uq_cref(self, obj)

    def pass_valu(self, obj):
        obj.mtxt = obj.mtxt + "pass_valu"
        return m.ReferencePassingTest.pass_valu(self, obj)

    def pass_mref(self, obj):
        obj.mtxt = obj.mtxt + "pass_mref"
        return m.ReferencePassingTest.pass_mref(self, obj)

    def pass_mptr(self, obj):
        obj.mtxt = obj.mtxt + "pass_mptr"
        return m.ReferencePassingTest.pass_mptr(self, obj)

    def pass_cref(self, obj):
        with pytest.raises(Exception):  # should be forbidden
            obj.mtxt = obj.mtxt + "pass_cref"
        return m.ReferencePassingTest.pass_cref(self, obj)

    def pass_cptr(self, obj):
        with pytest.raises(Exception):  # should be forbidden
            obj.mtxt = obj.mtxt + "pass_cptr"
        return m.ReferencePassingTest.pass_cptr(self, obj)


# roundtrip tests, creating a Passenger object in C++ that is passed by reference
# to a virtual method of a class derived in Python (PyReferencePassingTest1).
# If the object is correctly passed by reference, modifications should be visible
# by the C++ caller. The final obj's mtxt is returned by the check_* functions
# and validated here. Expected scheme: <func name>_[REF|COPY]
@pytest.mark.parametrize(
    "f, expected",
    [
        (m.check_roundtrip_uq_cref, "pass_uq_cref_REF"),
        (m.check_roundtrip_valu, "_COPY"),  # modification not passed back to C++
        (m.check_roundtrip_mref, "pass_mref_REF"),
        (m.check_roundtrip_mptr, "pass_mptr_REF"),
    ],
)
def test_refpassing1_roundtrip_modifyable(f, expected):
    c = PyReferencePassingTest1()
    assert f(c) == expected


@pytest.mark.parametrize(
    "f, expected",
    [
        # object passed as reference, but not modified
        (m.check_roundtrip_cref, "_REF"),
        (m.check_roundtrip_cptr, "_REF"),
    ],
)
# PYPY always copies the argument (to ensure constness?)
@pytest.mark.skipif("env.PYPY")
@pytest.mark.xfail  # maintaining constness isn't implemented yet
def test_refpassing1_roundtrip_const(f, expected):
    c = PyReferencePassingTest1()
    assert f(c) == expected


# Similar test as above, but now using PyReferencePassingTest2, calling
# to the C++ base class methods as well.
# Expected mtxt scheme: <func name>_MODIFIED_[REF|COPY]
@pytest.mark.parametrize(
    "f, expected",
    [
        pytest.param(  # cannot (yet) load not owned const unique_ptr& (for 2nd call)
            m.check_roundtrip_uq_cref,
            "pass_uq_cref",
            marks=pytest.mark.xfail,
        ),
        # object copied, modification not passed back to C++
        (m.check_roundtrip_valu, "_COPY"),
        (m.check_roundtrip_mref, "pass_mref_MODIFIED_REF"),
        (m.check_roundtrip_mptr, "pass_mptr_MODIFIED_REF"),
    ],
)
def test_refpassing2_roundtrip_modifyable(f, expected):
    c = PyReferencePassingTest2()
    assert f(c) == expected


@pytest.mark.parametrize(
    "f, expected",
    [
        # object passed as reference, but not modified
        (m.check_roundtrip_cref, "_REF"),
        (m.check_roundtrip_cptr, "_REF"),
    ],
)
# PYPY always copies the argument (to ensure constness?)
@pytest.mark.skipif("env.PYPY")
@pytest.mark.xfail  # maintaining constness isn't implemented yet
def test_refpassing2_roundtrip_const(f, expected):
    c = PyReferencePassingTest2()
    assert f(c) == expected
