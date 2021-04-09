# -*- coding: utf-8 -*-

import pybind11_tests.class_sh_trampoline_unique_ptr as m


class MyClass(m.Class):
    def foo(self):
        return 10

    def clone(self):
        return MyClass()


def test_py_clone_and_foo():
    obj = MyClass()
    assert obj.foo() == 10
    assert m.clone_and_foo(obj) == 10
