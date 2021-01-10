# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import type_caster_bare_interface_demo as m


def test_cast():
    assert m.rtrn_mpty_valu() == "cast_rref"
    assert m.rtrn_mpty_rref() == "cast_rref"
    assert m.rtrn_mpty_cref() == "cast_cref"
    assert m.rtrn_mpty_mref() == "cast_mref"
    assert m.rtrn_mpty_cptr() == "cast_cptr"
    assert m.rtrn_mpty_mptr() == "cast_mptr"


def test_load():
    assert m.pass_mpty_valu(None) == "load_valu"
    assert m.pass_mpty_rref(None) == "load_rref"
    assert m.pass_mpty_cref(None) == "load_cref"
    assert m.pass_mpty_mref(None) == "load_mref"
    assert m.pass_mpty_cptr(None) == "load_cptr"
    assert m.pass_mpty_mptr(None) == "load_mptr"


def test_cast_shared_ptr():
    assert m.rtrn_mpty_shmp() == "cast_shmp"
    assert m.rtrn_mpty_shcp() == "cast_shcp"


def test_load_shared_ptr():
    assert m.pass_mpty_shmp(None) == "load_shmp"
    assert m.pass_mpty_shcp(None) == "load_shcp"
