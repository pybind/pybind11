# -*- coding: utf-8 -*-
# Importing re before pytest after observing a PyPy CI flake when importing pytest first.
import re

import pytest

from pybind11_tests import class_sh_basic as m


def test_atyp_constructors():
    obj = m.atyp()
    assert obj.__class__.__name__ == "atyp"
    obj = m.atyp("")
    assert obj.__class__.__name__ == "atyp"
    obj = m.atyp("txtm")
    assert obj.__class__.__name__ == "atyp"


def check_regex(expected, actual):
    result = re.match(expected + "$", actual)
    if result is None:
        pytest.fail("expected: '{}' != actual: '{}'".format(expected, actual))


@pytest.mark.parametrize(
    "rtrn_f, expected",
    [
        (m.rtrn_valu, "rtrn_valu(_MvCtor){1,3}"),
        (m.rtrn_rref, "rtrn_rref(_MvCtor){1}"),
        (m.rtrn_cref, "rtrn_cref_CpCtor"),
        (m.rtrn_mref, "rtrn_mref"),
        (m.rtrn_cptr, "rtrn_cptr"),
        (m.rtrn_mptr, "rtrn_mptr"),
        (m.rtrn_shmp, "rtrn_shmp"),
        (m.rtrn_shcp, "rtrn_shcp"),
        (m.rtrn_uqmp, "rtrn_uqmp"),
        (m.rtrn_uqcp, "rtrn_uqcp"),
        (m.rtrn_udmp, "rtrn_udmp"),
        (m.rtrn_udcp, "rtrn_udcp"),
    ],
)
def test_cast(rtrn_f, expected):
    check_regex(expected, m.get_mtxt(rtrn_f()))


@pytest.mark.parametrize(
    "pass_f, mtxt, expected",
    [
        (m.pass_valu, "Valu", "pass_valu:Valu(_MvCtor){1,2}_CpCtor"),
        (m.pass_cref, "Cref", "pass_cref:Cref(_MvCtor){1,2}"),
        (m.pass_mref, "Mref", "pass_mref:Mref(_MvCtor){1,2}"),
        (m.pass_cptr, "Cptr", "pass_cptr:Cptr(_MvCtor){1,2}"),
        (m.pass_mptr, "Mptr", "pass_mptr:Mptr(_MvCtor){1,2}"),
        (m.pass_shmp, "Shmp", "pass_shmp:Shmp(_MvCtor){1,2}"),
        (m.pass_shcp, "Shcp", "pass_shcp:Shcp(_MvCtor){1,2}"),
        (m.pass_uqmp, "Uqmp", "pass_uqmp:Uqmp(_MvCtor){1,2}"),
        (m.pass_uqcp, "Uqcp", "pass_uqcp:Uqcp(_MvCtor){1,2}"),
    ],
)
def test_load_with_mtxt(pass_f, mtxt, expected):
    check_regex(expected, pass_f(m.atyp(mtxt)))


@pytest.mark.parametrize(
    "pass_f, rtrn_f, expected",
    [
        (m.pass_udmp, m.rtrn_udmp, "pass_udmp:rtrn_udmp"),
        (m.pass_udcp, m.rtrn_udcp, "pass_udcp:rtrn_udcp"),
    ],
)
def test_load_with_rtrn_f(pass_f, rtrn_f, expected):
    assert pass_f(rtrn_f()) == expected


@pytest.mark.parametrize(
    "pass_f, rtrn_f, expected",
    [
        (m.pass_uqmp, m.rtrn_uqmp, "pass_uqmp:rtrn_uqmp"),
        (m.pass_uqcp, m.rtrn_uqcp, "pass_uqcp:rtrn_uqcp"),
        (m.pass_udmp, m.rtrn_udmp, "pass_udmp:rtrn_udmp"),
        (m.pass_udcp, m.rtrn_udcp, "pass_udcp:rtrn_udcp"),
    ],
)
def test_pass_unique_ptr_disowns(pass_f, rtrn_f, expected):
    obj = rtrn_f()
    assert pass_f(obj) == expected
    with pytest.raises(ValueError) as exc_info:
        pass_f(obj)
    assert str(exc_info.value) == (
        "Missing value for wrapped C++ type: Python instance was disowned."
    )


@pytest.mark.parametrize(
    "pass_f, rtrn_f",
    [
        (m.pass_uqmp, m.rtrn_uqmp),
        (m.pass_uqcp, m.rtrn_uqcp),
        (m.pass_udmp, m.rtrn_udmp),
        (m.pass_udcp, m.rtrn_udcp),
    ],
)
def test_cannot_disown_use_count_ne_1(pass_f, rtrn_f):
    obj = rtrn_f()
    stash = m.SharedPtrStash()
    stash.Add(obj)
    with pytest.raises(ValueError) as exc_info:
        pass_f(obj)
    assert str(exc_info.value) == (
        "Cannot disown use_count != 1 (loaded_as_unique_ptr)."
    )


def test_unique_ptr_roundtrip(num_round_trips=1000):
    # Multiple roundtrips to stress-test instance registration/deregistration.
    recycled = m.atyp("passenger")
    for _ in range(num_round_trips):
        id_orig = id(recycled)
        recycled = m.unique_ptr_roundtrip(recycled)
        check_regex("passenger(_MvCtor){1,2}", m.get_mtxt(recycled))
        id_rtrn = id(recycled)
        # Ensure the returned object is a different Python instance.
        assert id_rtrn != id_orig
        id_orig = id_rtrn


# Validate moving an object from Python into a C++ object store
@pytest.mark.parametrize("pass_f", [m.store.pass_uq_valu, m.store.pass_uq_rref])
def test_unique_ptr_moved(pass_f):
    store = m.store()
    orig = m.atyp("O")
    mtxt_orig = m.get_mtxt(orig)
    ptr_orig = m.get_ptr(orig)
    assert re.match("O(_MvCtor){1,2}", mtxt_orig)

    pass_f(store, orig)  # pass object to C++ store c
    with pytest.raises(ValueError) as excinfo:
        m.get_mtxt(orig)
    assert "Python instance was disowned" in str(excinfo.value)

    del orig
    recycled = store.rtrn_uq_cref()
    assert m.get_ptr(recycled) == ptr_orig  # underlying C++ object doesn't change
    assert m.get_mtxt(recycled) == mtxt_orig  # object was not moved or copied


# This series of roundtrip tests checks how an object instance moved from
# Python to C++ (into store) can be later returned back to Python.
@pytest.mark.parametrize(
    "rtrn_f, moved_in",
    [
        (m.store.rtrn_uq_valu, True),  # moved back in
        (m.store.rtrn_uq_rref, True),  # moved back in
        (m.store.rtrn_uq_mref, None),  # forbidden
        (m.store.rtrn_uq_cref, False),  # fetched by reference
        (m.store.rtrn_mref, None),  # forbidden
        (m.store.rtrn_cref, False),  # fetched by reference
        (m.store.rtrn_mptr, None),  # forbidden
        (m.store.rtrn_cptr, False),  # fetched by reference
    ],
)
def test_unique_ptr_store_roundtrip(rtrn_f, moved_in):
    c = m.store()
    orig = m.atyp("passenger")
    ptr_orig = m.get_ptr(orig)

    c.pass_uq_valu(orig)  # pass object to C++ store c
    try:
        recycled = rtrn_f(c)  # retrieve object back from C++
    except RuntimeError:  # expect failure for rtrn_uq_lref
        assert moved_in is None
        return

    assert m.get_ptr(recycled) == ptr_orig  # do we yield the same object?
    if moved_in:  # store should have given up ownership?
        assert c.valid() is False
    else:  # store still helds the object
        assert c.valid() is True
        del recycled
        assert c.valid() is True


# Additionally to the above test_unique_ptr_store_roundtrip, this test
# validates that an object initially moved from Python to C++ can be returned
# to Python as a *const* reference/raw pointer/unique_ptr *and*, subsequently,
# passed from Python to C++ again. There shouldn't be any copy or move operation
# involved (We want the object to be passed by reference!)
@pytest.mark.parametrize(
    "rtrn_f",
    [m.store.rtrn_uq_cref, m.store.rtrn_cref, m.store.rtrn_cptr],
)
@pytest.mark.parametrize(
    "pass_f",
    [
        # This fails with: ValueError: Cannot disown non-owning holder (loaded_as_unique_ptr).
        # This could, at most, work for the combination rtrn_uq_cref() + pass_uq_cref(),
        # i.e. fetching a unique_ptr const-ref from C++ and passing the very same reference back.
        # Currently, it is forbidden - by design - to pass a unique_ptr const-ref to C++.
        # unique_ptrs are always moved (if possible).
        # To allow this use case, smart_holder would need to store the unique_ptr reference,
        # originally received from C++, e.g. using a union of unique_ptr + shared_ptr.
        pytest.param(m.store.pass_uq_cref, marks=pytest.mark.xfail),
        m.store.pass_cptr,
        m.store.pass_cref,
    ],
)
def test_unique_ptr_cref_store_roundtrip(rtrn_f, pass_f):
    c = m.store()
    passenger = m.atyp("passenger")
    mtxt_orig = m.get_mtxt(passenger)
    ptr_orig = m.get_ptr(passenger)

    # moves passenger to C++ (checked in test_unique_ptr_store_roundtrip)
    c.pass_uq_valu(passenger)

    for _ in range(10):
        cref = rtrn_f(c)  # fetches const reference, should keep-alive parent c
        assert pass_f(c, cref) == mtxt_orig  # no copy/move happened?
        assert m.get_ptr(cref) == ptr_orig  # it's still the same raw pointer


def test_py_type_handle_of_atyp():
    obj = m.py_type_handle_of_atyp()
    assert obj.__class__.__name__ == "pybind11_type"
