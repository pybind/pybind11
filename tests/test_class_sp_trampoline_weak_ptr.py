from __future__ import annotations

import gc

import pytest

import env
import pybind11_tests.class_sp_trampoline_weak_ptr as m


class PyDrvd(m.VirtBase):
    def get_code(self):
        return 200


@pytest.mark.parametrize(("vtype", "expected_code"), [(m.VirtBase, 100), (PyDrvd, 200)])
def test_with_wp_owner(vtype, expected_code):
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


@pytest.mark.parametrize(("vtype", "expected_code"), [(m.VirtBase, 100), (PyDrvd, 200)])
def test_with_sp_owner(vtype, expected_code):
    spo = m.SpOwner()
    assert spo.get_code() == -888

    obj = vtype()
    assert obj.get_code() == expected_code

    spo.set_sp(obj)
    assert spo.get_code() == expected_code

    del obj
    if env.PYPY or env.GRAALPY:
        pytest.skip("Cannot reliably trigger GC")
    print("\nLOOOK BEFORE spo.get_code() AFTER del obj", flush=True)
    assert spo.get_code() == 100  # Inheritance slicing (issue #1333)
    print("\nLOOOK  AFTER spo.get_code() AFTER del obj", flush=True)


@pytest.mark.parametrize(("vtype", "expected_code"), [(m.VirtBase, 100), (PyDrvd, 200)])
def test_with_sp_and_wp_owners(vtype, expected_code):
    spo = m.SpOwner()
    wpo = m.WpOwner()

    obj = vtype()
    spo.set_sp(obj)
    wpo.set_wp(obj)

    assert spo.get_code() == expected_code
    assert wpo.get_code() == expected_code

    del obj
    if env.PYPY or env.GRAALPY:
        pytest.skip("Cannot reliably trigger GC")

    # Inheritance slicing (issue #1333)
    assert spo.get_code() == 100
    assert wpo.get_code() == 100

    del spo
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
