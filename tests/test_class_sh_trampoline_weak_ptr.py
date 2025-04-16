from __future__ import annotations

import pytest

import env  # noqa: F401
import pybind11_tests.class_sh_trampoline_weak_ptr as m


@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
def test_weak_ptr_base():
    tester = m.WpBaseTester()

    obj = m.WpBase()

    tester.set_object(obj)

    assert tester.is_expired() is False
    assert tester.is_base_used() is True
    assert tester.get_object().is_base_used() is True


@pytest.mark.skipif("env.GRAALPY", reason="Cannot reliably trigger GC")
def test_weak_ptr_child():
    class PyChild(m.WpBase):
        def is_base_used(self):
            return False

    tester = m.WpBaseTester()

    obj = PyChild()

    tester.set_object(obj)

    assert tester.is_expired() is False
    assert tester.is_base_used() is False
    assert tester.get_object().is_base_used() is False
