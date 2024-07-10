from __future__ import annotations

import env  # noqa: F401
from pybind11_tests import class_sh_disowning_mi as m


def test_disown_d():
    d = m.D()
    m.disown_b(d)


def test_shptr_copy():
    m.test_ShPtr_copy()
