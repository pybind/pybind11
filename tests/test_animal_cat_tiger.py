from __future__ import annotations

import pytest

from pybind11_tests import class_animal as m


@pytest.mark.parametrize("tiger_type", [m.TigerSP, m.TigerSH])
def test_clone(tiger_type):
    tiger = tiger_type()
    cloned = tiger.clone()
    assert isinstance(cloned, tiger_type)
