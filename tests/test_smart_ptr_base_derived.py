# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import smart_ptr_base_derived as m

CBASE_GET_INT_RESULT = 90146438
CDERIVED_GET_INT_RESULT = 31607978
VDERIVED_GET_INT_RESULT = 29852452

def test_concrete():
    m.to_cout("")
    m.to_cout("")
    m.to_cout("make_shared_cderived")
    cd = m.make_shared_cderived()
    assert cd.get_int() == CDERIVED_GET_INT_RESULT
    m.pass_shared_cderived(cd)
    m.pass_shared_cbase(cd)
    cb = m.make_shared_cderived_up_cast()
    assert cb.get_int() == CBASE_GET_INT_RESULT
    m.pass_shared_cbase(cb)
    with pytest.raises(TypeError):
        m.pass_shared_cderived(cb)
    m.to_cout("")

def test_virtual():
    m.to_cout("")
    m.to_cout("")
    m.to_cout("make_shared_vderived")
    vd = m.make_shared_vderived()
    assert vd.get_int() == VDERIVED_GET_INT_RESULT
    m.pass_shared_vderived(vd)
    m.pass_shared_vbase(vd)
    vd_uc = m.make_shared_vderived_up_cast()
    assert vd_uc.get_int() == VDERIVED_GET_INT_RESULT
    assert isinstance(vd_uc, m.vderived)  # pybind11 un-did upcast.
    m.pass_shared_vbase(vd_uc)
    m.pass_shared_vderived(vd_uc)
    with pytest.raises(TypeError):
        m.pass_shared_vrederived(vd_uc)
    m.to_cout("")
