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


def test_cast():
    assert m.get_mtxt(m.rtrn_valu_atyp()) == "rtrn_valu"
    assert m.get_mtxt(m.rtrn_rref_atyp()) == "rtrn_rref"
    assert m.get_mtxt(m.rtrn_cref_atyp()) == "rtrn_cref"
    assert m.get_mtxt(m.rtrn_mref_atyp()) == "rtrn_mref"
    assert m.get_mtxt(m.rtrn_cptr_atyp()) == "rtrn_cptr"
    assert m.get_mtxt(m.rtrn_mptr_atyp()) == "rtrn_mptr"


def test_load():
    assert m.pass_valu_atyp(m.atyp("Valu")) == "pass_valu:Valu"
    assert m.pass_rref_atyp(m.atyp("Rref")) == "pass_rref:Rref"
    assert m.pass_cref_atyp(m.atyp("Cref")) == "pass_cref:Cref"
    assert m.pass_mref_atyp(m.atyp("Mref")) == "pass_mref:Mref"
    assert m.pass_cptr_atyp(m.atyp("Cptr")) == "pass_cptr:Cptr"
    assert m.pass_mptr_atyp(m.atyp("Mptr")) == "pass_mptr:Mptr"


def test_cast_shared_ptr():
    assert m.get_mtxt(m.rtrn_shmp_atyp()) == "rtrn_shmp"
    assert m.get_mtxt(m.rtrn_shcp_atyp()) == "rtrn_shcp"


def test_load_shared_ptr():
    assert m.pass_shmp_atyp(m.atyp("Shmp")) == "pass_shmp:Shmp"
    assert m.pass_shcp_atyp(m.atyp("Shcp")) == "pass_shcp:Shcp"


def test_cast_unique_ptr():
    assert m.get_mtxt(m.rtrn_uqmp_atyp()) == "rtrn_uqmp"
    assert m.get_mtxt(m.rtrn_uqcp_atyp()) == "rtrn_uqcp"


def test_load_unique_ptr():
    assert m.pass_uqmp_atyp(m.atyp("Uqmp")) == "pass_uqmp:Uqmp"
    assert m.pass_uqcp_atyp(m.atyp("Uqcp")) == "pass_uqcp:Uqcp"


@pytest.mark.parametrize(
    "pass_atyp, argm, rtrn",
    [
        (m.pass_uqmp_atyp, "Uqmp", "pass_uqmp:Uqmp"),
        (m.pass_uqcp_atyp, "Uqcp", "pass_uqcp:Uqcp"),
    ],
)
def test_pass_unique_ptr_disowns(pass_atyp, argm, rtrn):
    obj = m.atyp(argm)
    assert pass_atyp(obj) == rtrn
    with pytest.raises(RuntimeError) as exc_info:
        m.pass_uqmp_atyp(obj)
    assert str(exc_info.value) == (
        "Missing value for wrapped C++ type:"
        " Python instance is uninitialized or was disowned."
    )


def test_unique_ptr_roundtrip(num_round_trips=1000):
    # Multiple roundtrips to stress-test instance registration/deregistration.
    recycled = m.atyp("passenger")
    for _ in range(num_round_trips):
        id_orig = id(recycled)
        recycled = m.unique_ptr_roundtrip(recycled)
        assert m.get_mtxt(recycled) == "passenger"
        id_rtrn = id(recycled)
        # Ensure the returned object is a different Python instance.
        assert id_rtrn != id_orig
        id_orig = id_rtrn


def test_py_type_handle_of_atyp():
    obj = m.py_type_handle_of_atyp()
    assert obj.__class__.__name__ == "pybind11_type"
