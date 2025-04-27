from __future__ import annotations

import gc
import weakref

import pytest

import env
import pybind11_tests.potentially_slicing_shared_ptr as m


class PyDrvd(m.VirtBase):
    def get_code(self):
        return 200


@pytest.mark.parametrize(("vtype", "expected_code"), [(m.VirtBase, 100), (PyDrvd, 200)])
@pytest.mark.parametrize(
    "rtrn_meth", ["rtrn_obj_cast_shared_ptr", "rtrn_potentially_slicing_shared_ptr"]
)
def test_rtrn_obj_cast_shared_ptr(vtype, rtrn_meth, expected_code):
    obj = vtype()
    ptr = getattr(m, rtrn_meth)(obj)
    assert ptr.get_code() == expected_code
    objref = weakref.ref(obj)
    del obj
    gc.collect()
    assert ptr.get_code() == expected_code  # the ptr Python object keeps obj alive
    assert objref() is not None
    del ptr
    gc.collect()
    assert objref() is None


@pytest.mark.parametrize(("vtype", "expected_code"), [(m.VirtBase, 100), (PyDrvd, 200)])
def test_with_sp_owner(vtype, expected_code):
    spo = m.SpOwner()
    assert spo.get_code() == -888

    obj = vtype()
    assert obj.get_code() == expected_code

    spo.set_sp(obj)
    assert spo.get_code() == expected_code

    del obj
    gc.collect()
    assert spo.get_code() == expected_code


@pytest.mark.parametrize(("vtype", "expected_code"), [(m.VirtBase, 100), (PyDrvd, 200)])
@pytest.mark.parametrize("set_meth", ["set_wp", "set_wp_potentially_slicing"])
def test_with_wp_owner(vtype, set_meth, expected_code):
    wpo = m.WpOwner()
    assert wpo.get_code() == -999

    obj = vtype()
    assert obj.get_code() == expected_code

    getattr(wpo, set_meth)(obj)
    if vtype is m.VirtBase or set_meth == "set_wp_potentially_slicing":
        assert wpo.get_code() == expected_code
    else:
        assert wpo.get_code() == -999  # see PR #5624

    del obj
    gc.collect()
    if not (env.PYPY or env.GRAALPY):
        assert wpo.get_code() == -999
