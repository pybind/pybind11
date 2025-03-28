from __future__ import annotations

import pytest

from pybind11_tests import scipy_low_level_callable as m


def test_square():
    assert m.square(2.0) == 4.0


def test_get_capsule_for_scipy_LowLevelCallable():
    cap = m.square.__self__.get_capsule_for_scipy_LowLevelCallable(
        signature="double (double)"
    )
    assert repr(cap).startswith('<capsule object "double (double)" at 0x')


def test_with_scipy_LowLevelCallable():
    scipy = pytest.importorskip("scipy")
    # Explicit import needed with some (older) scipy versions:
    from scipy import integrate

    llc = scipy.LowLevelCallable(
        m.square.__self__.get_capsule_for_scipy_LowLevelCallable(
            signature="double (double)"
        )
    )
    integral = integrate.quad(llc, 0, 1)
    assert integral[0] == pytest.approx(1 / 3, rel=1e-12)
