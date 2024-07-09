from __future__ import annotations

import pytest

from pybind11_tests import wip as m


def test_mixed():
    obj1a = m.Atype1(90)
    obj2a = m.Atype2(25)
    obj1b = m.Atype1(0)
    obj2b = m.Atype2(0)

    print("\nLOOOK A BEFORE m.mixed(obj1a, obj2a)", flush=True)
    assert m.mixed(obj1a, obj2a) == (90 * 10 + 1) * 200 + (25 * 10 + 2) * 20
    print("\nLOOOK A  AFTER m.mixed(obj1a, obj2a)", flush=True)

    print("\nLOOOK B BEFORE m.mixed(obj1b, obj2a)", flush=True)
    with pytest.raises(ValueError):
        m.mixed(obj1b, obj2a)
    print("\nLOOOK B  AFTER m.mixed(obj1b, obj2a)", flush=True)

    print("\nLOOOK C BEFORE m.mixed(obj1a, obj2b)", flush=True)
    with pytest.raises(ValueError):
        m.mixed(obj1a, obj2b)
    print("\nLOOOK C  AFTER m.mixed(obj1a, obj2b)", flush=True)
