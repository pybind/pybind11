# -*- coding: utf-8 -*-

from pybind11_tests import classh_mock as m


def test_foobar():
    # Not really testing anything in particular. The main purpose of this test is to ensure the
    # suggested BOILERPLATE code block in test_classh_mock.cpp is correct.
    assert m.Foo0()
    assert m.Foo1()
    assert m.Foo2()
