from __future__ import annotations

from pybind11_tests import class_sh_unique_ptr_custom_deleter as m


def test_create():
    pet = m.create("abc")
    assert pet.name == "abc"
