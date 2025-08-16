from __future__ import annotations

from pybind11_tests import class_animal as m


def test_animals():
    tiger = m.Tiger()
    tiger.clone()
