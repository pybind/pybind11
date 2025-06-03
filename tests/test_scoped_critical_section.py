from __future__ import annotations

import pytest

from pybind11_tests import scoped_critical_section as m


@pytest.mark.skipif(not m.has_barrier, reason="no <barrier>")
def test_scoped_critical_section() -> None:
    for _ in range(64):
        m.test_scoped_critical_section()


@pytest.mark.skipif(not m.has_barrier, reason="no <barrier>")
def test_scoped_critical_section2() -> None:
    for _ in range(64):
        m.test_scoped_critical_section2()


@pytest.mark.skipif(not m.has_barrier, reason="no <barrier>")
def test_scoped_critical_section2_same_object_no_deadlock() -> None:
    for _ in range(64):
        m.test_scoped_critical_section2_same_object_no_deadlock()
