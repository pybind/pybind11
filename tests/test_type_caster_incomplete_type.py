from __future__ import annotations

from pybind11_tests import type_caster_incomplete_type as m


def test_rtrn_fwd_decl_type_ptr():
    assert m.rtrn_fwd_decl_type_ptr() is None


def test_pass_fwd_decl_type_ptr():
    assert m.pass_fwd_decl_type_ptr(None) is None
