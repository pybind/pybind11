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
    assert m.rtrn_mpty_valu() == "cast_rref"
    assert m.rtrn_mpty_rref() == "cast_rref"
    assert m.rtrn_mpty_cref() == "cast_cref"
    assert m.rtrn_mpty_mref() == "cast_mref"
    assert m.rtrn_mpty_cptr() == "cast_cptr"
    assert m.rtrn_mpty_mptr() == "cast_mptr"


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
    assert m.pass_mpty_shmp(m.mpty()) == "load_shmp"
    assert m.pass_mpty_shcp(m.mpty()) == "load_shcp"


def test_cast_unique_ptr():
    assert m.rtrn_mpty_uqmp() == "cast_uqmp"
    assert m.rtrn_mpty_uqcp() == "cast_uqcp"


def test_load_unique_ptr():
    assert m.pass_mpty_uqmp(m.mpty()) == "load_uqmp"
    assert m.pass_mpty_uqcp(m.mpty()) == "load_uqcp"
