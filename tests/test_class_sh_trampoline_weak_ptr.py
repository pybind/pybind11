from __future__ import annotations

import gc

import pytest

import env
import pybind11_tests.class_sh_trampoline_weak_ptr as m


class PyDrvd(m.VirtBase):
    def get_code(self):
        return 200


@pytest.mark.parametrize(("vtype", "expected_code"), [(m.VirtBase, 100), (PyDrvd, 200)])
def test_weak_ptr_owner(vtype, expected_code):
    wpo = m.WpOwner()
    assert wpo.get_code() == -999

    obj = vtype()
    assert obj.get_code() == expected_code

    wpo.set_wp(obj)
    if vtype is m.VirtBase:
        assert wpo.get_code() == expected_code
    else:
        assert wpo.get_code() == -999  # THIS NEEDS FIXING (issue #5623)

    del obj
    if env.PYPY or env.GRAALPY:
        pytest.skip("Cannot reliably trigger GC")
    assert wpo.get_code() == -999


@pytest.mark.parametrize(("vtype", "expected_code"), [(m.VirtBase, 100), (PyDrvd, 200)])
def test_pass_through_sp_VirtBase(vtype, expected_code):
    obj = vtype()
    ptr = m.pass_through_sp_VirtBase(obj)
    print("\nLOOOK BEFORE del obj", flush=True)
    del obj
    print("\nLOOOK  AFTER del obj", flush=True)
    gc.collect()
    print("\nLOOOK  AFTER gc.collect()", flush=True)
    assert ptr.get_code() == expected_code
    print("\nLOOOK  AFTER ptr.get_code()", flush=True)
