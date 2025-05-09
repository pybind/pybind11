# Copyright (c) 2025 The pybind Community.
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

# This module tests the interaction of pybind11's shared_ptr and smart_holder
# mechanisms with trampoline object lifetime management and inheritance slicing.
#
# The following combinations are covered:
#
# - Holder type: std::shared_ptr (class_ holder) vs.
#                py::smart_holder
# - Conversion function: obj.cast<std::shared_ptr<T>>() vs.
#                        py::potentially_slicing_weak_ptr<T>(obj)
# - Python object type: C++ base class vs.
#                       Python-derived trampoline class
#
# The tests verify
#
# - that casting or passing Python objects into functions returns usable
#   std::shared_ptr<T> instances.
# - that inheritance slicing occurs as expected in controlled cases
#   (issue #1333).
# - that surprising weak_ptr behavior (issue #5623) can be reproduced when
#   smart_holder is used.
# - that the trampoline object remains alive in all situations
#   (no use-after-free) as long as the C++ shared_ptr exists.
#
# Where applicable, trampoline state is introspected to confirm whether the
# C++ object retains knowledge of the Python override or has fallen back to
# the base implementation.

from __future__ import annotations

import gc
import weakref

import pytest

import env
import pybind11_tests.potentially_slicing_weak_ptr as m


class PyDrvdSH(m.VirtBaseSH):
    def get_code(self):
        return 200


class PyDrvdSP(m.VirtBaseSP):
    def get_code(self):
        return 200


VIRT_BASE_TYPES = {
    "SH": {100: m.VirtBaseSH, 200: PyDrvdSH},
    "SP": {100: m.VirtBaseSP, 200: PyDrvdSP},
}

RTRN_FUNCS = {
    "SH": {
        "oc": m.SH_rtrn_obj_cast_shared_ptr,
        "ps": m.SH_rtrn_potentially_slicing_shared_ptr,
    },
    "SP": {
        "oc": m.SP_rtrn_obj_cast_shared_ptr,
        "ps": m.SP_rtrn_potentially_slicing_shared_ptr,
    },
}

SP_OWNER_TYPES = {
    "SH": m.SH_SpOwner,
    "SP": m.SP_SpOwner,
}

WP_OWNER_TYPES = {
    "SH": m.SH_WpOwner,
    "SP": m.SP_WpOwner,
}

GC_IS_RELIABLE = not (env.PYPY or env.GRAALPY)


@pytest.mark.parametrize("expected_code", [100, 200])
@pytest.mark.parametrize("rtrn_kind", ["oc", "ps"])
@pytest.mark.parametrize("holder_kind", ["SH", "SP"])
def test_rtrn_obj_cast_shared_ptr(holder_kind, rtrn_kind, expected_code):
    obj = VIRT_BASE_TYPES[holder_kind][expected_code]()
    ptr = RTRN_FUNCS[holder_kind][rtrn_kind](obj)
    assert ptr.get_code() == expected_code
    objref = weakref.ref(obj)
    del obj
    gc.collect()
    assert ptr.get_code() == expected_code  # the ptr Python object keeps obj alive
    assert objref() is not None
    del ptr
    gc.collect()
    if GC_IS_RELIABLE:
        assert objref() is None


@pytest.mark.parametrize("expected_code", [100, 200])
@pytest.mark.parametrize("holder_kind", ["SH", "SP"])
def test_with_sp_owner(holder_kind, expected_code):
    spo = SP_OWNER_TYPES[holder_kind]()
    assert spo.get_code() == -888
    assert spo.get_trampoline_state() == "sp nullptr"

    obj = VIRT_BASE_TYPES[holder_kind][expected_code]()
    assert obj.get_code() == expected_code

    spo.set_sp(obj)
    assert spo.get_code() == expected_code
    expected_trampoline_state = (
        "dynamic_cast failed" if expected_code == 100 else "trampoline alive"
    )
    assert spo.get_trampoline_state() == expected_trampoline_state

    del obj
    gc.collect()
    if holder_kind == "SH":
        assert spo.get_code() == expected_code
    elif GC_IS_RELIABLE:
        assert (
            spo.get_code() == 100
        )  # see issue #1333 (inheritance slicing) and PR #5624
    assert spo.get_trampoline_state() == expected_trampoline_state


@pytest.mark.parametrize("expected_code", [100, 200])
@pytest.mark.parametrize("set_meth", ["set_wp", "set_wp_potentially_slicing"])
@pytest.mark.parametrize("holder_kind", ["SH", "SP"])
def test_with_wp_owner(holder_kind, set_meth, expected_code):
    wpo = WP_OWNER_TYPES[holder_kind]()
    assert wpo.get_code() == -999
    assert wpo.get_trampoline_state() == "sp nullptr"

    obj = VIRT_BASE_TYPES[holder_kind][expected_code]()
    assert obj.get_code() == expected_code

    getattr(wpo, set_meth)(obj)
    if (
        holder_kind == "SP"
        or expected_code == 100
        or set_meth == "set_wp_potentially_slicing"
    ):
        assert wpo.get_code() == expected_code
    else:
        assert wpo.get_code() == -999  # see issue #5623 (weak_ptr expired) and PR #5624
    if expected_code == 100:
        expected_trampoline_state = "dynamic_cast failed"
    elif holder_kind == "SH" and set_meth == "set_wp":
        expected_trampoline_state = "sp nullptr"
    else:
        expected_trampoline_state = "trampoline alive"
    assert wpo.get_trampoline_state() == expected_trampoline_state

    del obj
    gc.collect()
    if GC_IS_RELIABLE:
        assert wpo.get_code() == -999


def test_potentially_slicing_weak_ptr_not_convertible_error():
    with pytest.raises(Exception) as excinfo:
        m.SH_rtrn_potentially_slicing_shared_ptr("")
    assert str(excinfo.value) == (
        '"str" object is not convertible to std::weak_ptr<T>'
        " (with T = pybind11_tests::potentially_slicing_weak_ptr::VirtBase<0>)"
    )
    with pytest.raises(Exception) as excinfo:
        m.SP_rtrn_potentially_slicing_shared_ptr([])
    assert str(excinfo.value) == (
        '"list" object is not convertible to std::weak_ptr<T>'
        " (with T = pybind11_tests::potentially_slicing_weak_ptr::VirtBase<1>)"
    )
