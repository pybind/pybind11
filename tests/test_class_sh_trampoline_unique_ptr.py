from __future__ import annotations

import pybind11_tests.class_sh_trampoline_unique_ptr as m


class MyClass(m.Class):
    def foo(self):
        return 10 + self.get_val()

    def clone(self):
        cloned = MyClass()
        cloned.set_val(self.get_val() + 3)
        return cloned


def test_m_clone():
    obj = MyClass()
    while True:
        obj.set_val(5)
        obj = m.clone(obj)
        assert obj.get_val() == 5 + 3
        assert obj.foo() == 10 + 5 + 3
        return  # Comment out for manual leak checking (use `top` command).


def test_m_clone_and_foo():
    obj = MyClass()
    obj.set_val(7)
    while True:
        assert m.clone_and_foo(obj) == 10 + 7 + 3
        return  # Comment out for manual leak checking (use `top` command).
