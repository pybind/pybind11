from __future__ import annotations

from pybind11_tests import wip as m


def test_doc():
    assert m.__doc__ == "WIP"


def test_some_type_ctor():
    obj = m.SomeType()
    assert isinstance(obj, m.SomeType)


def test_make_some_type():
    obj = m.make_some_type()
    assert isinstance(obj, m.SomeType)
