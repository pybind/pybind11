from __future__ import annotations

import pytest

import env
import pybind11_tests.class_sp_trampoline_weak_ptr as m


class PyDrvd(m.VirtBase):
    def get_code(self):
        return 200


@pytest.mark.parametrize(("vtype", "expected_code"), [(m.VirtBase, 100), (PyDrvd, 200)])
def test_weak_ptr_base(vtype, expected_code):
    wpo = m.WpOwner()
    assert wpo.get_code() == -999

    obj = vtype()
    assert obj.get_code() == expected_code

    wpo.set_wp(obj)
    assert wpo.get_code() == expected_code

    del obj
    if env.PYPY or env.GRAALPY:
        pytest.skip("Cannot reliably trigger GC")
    assert wpo.get_code() == -999
