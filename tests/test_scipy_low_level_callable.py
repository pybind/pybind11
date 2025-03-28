from __future__ import annotations

import pytest

from pybind11_tests import scipy_low_level_callable as m


def test_square():
    assert m.square(2.0) == 4.0


def _m_square_self():
    try:
        return m.square.__self__
    except AttributeError as e:
        pytest.skip(f"{str(e)}")


def test_get_capsule_for_scipy_LowLevelCallable():
    cap = _m_square_self().get_capsule_for_scipy_LowLevelCallable(
        signature="double (double)"
    )
    assert repr(cap).startswith("<capsule object ")


def test_with_scipy_LowLevelCallable():
    scipy = pytest.importorskip("scipy")
    # Explicit import needed with some (older) scipy versions:
    from scipy import integrate

    llc = scipy.LowLevelCallable(
        _m_square_self().get_capsule_for_scipy_LowLevelCallable(
            signature="double (double)"
        )
    )
    integral = integrate.quad(llc, 0, 1)
    assert integral[0] == pytest.approx(1 / 3, rel=1e-12)
