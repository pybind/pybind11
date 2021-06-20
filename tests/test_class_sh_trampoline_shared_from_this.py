# -*- coding: utf-8 -*-

import pybind11_tests.class_sh_trampoline_shared_from_this as m


class PyWithSft(m.WithSft):
    pass


def test_pass_shared_ptr():
    m.to_cout("")
    m.to_cout(">>> obj = PyWithSft()")
    obj = PyWithSft()
    m.to_cout(">>> m.pass_shared_ptr(obj) #1")
    m.pass_shared_ptr(obj)
    m.to_cout(">>> m.pass_shared_ptr(obj) #2")
    m.pass_shared_ptr(obj)
    m.to_cout(">>> del obj")
    del obj
