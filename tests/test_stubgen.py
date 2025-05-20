from __future__ import annotations

from pybind11_tests import stubgen as m


def test_stubgen() -> None:
    assert m.add_int(1, 2) == 3
