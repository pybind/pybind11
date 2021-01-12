# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import classh_wip as m


def test_mpty_constructors():
    e = m.mpty()
    assert e.__class__.__name__ == "mpty"
    e = m.mpty("")
    assert e.__class__.__name__ == "mpty"
    e = m.mpty("txtm")
    assert e.__class__.__name__ == "mpty"


def test_cast():
    assert m.get_mtxt(m.rtrn_mpty_valu()) == "rtrn_valu"
    #                                         rtrn_rref exercised separately.
    assert m.get_mtxt(m.rtrn_mpty_cref()) == "rtrn_cref"
    assert m.get_mtxt(m.rtrn_mpty_mref()) == "rtrn_mref"
    assert m.get_mtxt(m.rtrn_mpty_cptr()) == "rtrn_cptr"
    assert m.get_mtxt(m.rtrn_mpty_mptr()) == "rtrn_mptr"


def test_cast_rref():
    e = m.rtrn_mpty_rref()
    assert e.__class__.__name__ == "mpty"
    with pytest.raises(RuntimeError):
        m.get_mtxt(e)  # E.g. basic_string::_M_construct null not valid


def test_load():
    assert m.pass_mpty_valu(m.mpty("Valu")) == "pass_valu:Valu"
    assert m.pass_mpty_rref(m.mpty("Rref")) == "pass_rref:Rref"
    assert m.pass_mpty_cref(m.mpty("Cref")) == "pass_cref:Cref"
    assert m.pass_mpty_mref(m.mpty("Mref")) == "pass_mref:Mref"
    assert m.pass_mpty_cptr(m.mpty("Cptr")) == "pass_cptr:Cptr"
    assert m.pass_mpty_mptr(m.mpty("Mptr")) == "pass_mptr:Mptr"


def test_cast_shared_ptr():
    assert m.rtrn_mpty_shmp() == "cast_shmp"
    assert m.rtrn_mpty_shcp() == "cast_shcp"


def test_load_shared_ptr():
    assert m.pass_mpty_shmp(m.mpty("Shmp")) == "pass_shmp:Shmp"
    assert m.pass_mpty_shcp(m.mpty("Shcp")) == "pass_shcp:Shcp"


def test_cast_unique_ptr():
    assert m.rtrn_mpty_uqmp() == "cast_uqmp"
    assert m.rtrn_mpty_uqcp() == "cast_uqcp"


def test_load_unique_ptr():
    assert m.pass_mpty_uqmp(m.mpty("Uqmp")) == "pass_uqmp:Uqmp"
    assert m.pass_mpty_uqcp(m.mpty("Uqcp")) == "pass_uqcp:Uqcp"


@pytest.mark.parametrize(
    "pass_mpty, argm, rtrn",
    [
        (m.pass_mpty_uqmp, "Uqmp", "pass_uqmp:Uqmp"),
        (m.pass_mpty_uqcp, "Uqcp", "pass_uqcp:Uqcp"),
    ],
)
def test_pass_unique_ptr_disowns(pass_mpty, argm, rtrn):
    obj = m.mpty(argm)
    assert pass_mpty(obj) == rtrn
    with pytest.raises(RuntimeError) as exc_info:
        m.pass_mpty_uqmp(obj)
    assert str(exc_info.value) == "Cannot disown nullptr (as_unique_ptr)."
