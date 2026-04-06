from __future__ import annotations

from pybind11_tests import pybind11_pytorch_regressions as m


def test_pytorch_like_get_tracing_state_aliases_singleton_shared_ptr():
    a = m._get_tracing_state()
    b = m._get_tracing_state()

    a.value = 17

    assert b.value == 17
    assert m._get_tracing_state().value == 17


def test_pytorch_like_compilation_unit_get_interface_aliases_member_shared_ptr():
    cu = m.CompilationUnit()

    a = cu.get_interface("iface")
    b = cu.get_interface("iface")

    a.value = 23

    assert b.value == 23
    assert cu.get_interface("iface").value == 23
