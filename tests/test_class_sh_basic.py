# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import class_sh_basic as m


def test_atyp_constructors():
    obj = m.atyp()
    assert obj.__class__.__name__ == "atyp"
    obj = m.atyp("")
    assert obj.__class__.__name__ == "atyp"
    obj = m.atyp("txtm")
    assert obj.__class__.__name__ == "atyp"


@pytest.mark.parametrize(
    "rtrn_f, expected",
    [
        (m.rtrn_valu, "rtrn_valu.MvCtor"),
        (m.rtrn_rref, "rtrn_rref.MvCtor"),
        (m.rtrn_cref, "rtrn_cref.CpCtor"),
        (m.rtrn_mref, "rtrn_mref.CpCtor"),
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
    assert m.get_mtxt(rtrn_f()) == expected


@pytest.mark.parametrize(
    "pass_f, mtxt, expected",
    [
        (m.pass_valu, "Valu", "pass_valu:Valu.MvCtor.CpCtor"),
        (m.pass_rref, "Rref", "pass_rref:Rref.MvCtor.CpCtor"),
        (m.pass_cref, "Cref", "pass_cref:Cref.MvCtor"),
        (m.pass_mref, "Mref", "pass_mref:Mref.MvCtor"),
        (m.pass_cptr, "Cptr", "pass_cptr:Cptr.MvCtor"),
        (m.pass_mptr, "Mptr", "pass_mptr:Mptr.MvCtor"),
        (m.pass_shmp, "Shmp", "pass_shmp:Shmp.MvCtor"),
        (m.pass_shcp, "Shcp", "pass_shcp:Shcp.MvCtor"),
        (m.pass_uqmp, "Uqmp", "pass_uqmp:Uqmp.MvCtor"),
        (m.pass_uqcp, "Uqcp", "pass_uqcp:Uqcp.MvCtor"),
    ],
)
def test_load_with_mtxt(pass_f, mtxt, expected):
    assert pass_f(m.atyp(mtxt)) == expected


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
    with pytest.raises(RuntimeError) as exc_info:
        pass_f(obj)
    assert str(exc_info.value) == (
        "Missing value for wrapped C++ type: Python instance was disowned."
    )


def test_unique_ptr_roundtrip(num_round_trips=1000):
    # Multiple roundtrips to stress-test instance registration/deregistration.
    recycled = m.atyp("passenger")
    for _ in range(num_round_trips):
        id_orig = id(recycled)
        recycled = m.unique_ptr_roundtrip(recycled)
        assert m.get_mtxt(recycled) == "passenger.MvCtor"
        id_rtrn = id(recycled)
        # Ensure the returned object is a different Python instance.
        assert id_rtrn != id_orig
        id_orig = id_rtrn


def test_py_type_handle_of_atyp():
    obj = m.py_type_handle_of_atyp()
    assert obj.__class__.__name__ == "pybind11_type"
