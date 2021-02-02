# -*- coding: utf-8 -*-

from pybind11_tests import class_sh_factory_constructors as m


def test_atyp_factories():
    assert m.atyp_valu().get_mtxt() == "Valu"
    assert m.atyp_rref().get_mtxt() == "Rref"
    # sert m.atyp_cref().get_mtxt() == "Cref"
    # sert m.atyp_mref().get_mtxt() == "Mref"
    # sert m.atyp_cptr().get_mtxt() == "Cptr"
    assert m.atyp_mptr().get_mtxt() == "Mptr"
    assert m.atyp_shmp().get_mtxt() == "Shmp"
    # sert m.atyp_shcp().get_mtxt() == "Shcp"
    assert m.atyp_uqmp().get_mtxt() == "Uqmp"
    # sert m.atyp_uqcp().get_mtxt() == "Uqcp"
    assert m.atyp_udmp().get_mtxt() == "Udmp"
    # sert m.atyp_udcp().get_mtxt() == "Udcp"
