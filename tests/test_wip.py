from __future__ import annotations

from pybind11_tests import wip as m


def test_doc():
    assert m.__doc__ == "WIP"
