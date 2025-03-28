from __future__ import annotations

import pytest

import pybind11_tests
from pybind11_tests import scipy_low_level_callable as m


def test_square21():
    assert m.square21(2.0) == 2.0 * 2.0 * 21


def _m_square21_self():
    try:
        return m.square21.__self__
    except AttributeError as e:
        pytest.skip(f"{str(e)}")


def test_python_function_record_static_properties():
    func_rec = _m_square21_self()
    assert func_rec.PYBIND11_DETAIL_FUNCTION_RECORD_ABI_ID == "v1"
    assert func_rec.PYBIND11_PLATFORM_ABI_ID in pybind11_tests.PYBIND11_INTERNALS_ID


def test_get_capsule_for_scipy_LowLevelCallable():
    cap = (
        _m_square21_self().get_capsule_for_scipy_LowLevelCallable_NO_ABI_OR_TYPE_SAFETY(
            signature="double (double)"
        )
    )
    assert repr(cap).startswith("<capsule object ")


def test_with_scipy_LowLevelCallable():
    scipy = pytest.importorskip("scipy")
    # Explicit import needed with some (older) scipy versions:
    from scipy import integrate

    llc = scipy.LowLevelCallable(
        _m_square21_self().get_capsule_for_scipy_LowLevelCallable_NO_ABI_OR_TYPE_SAFETY(
            signature="double (double)"
        )
    )
    integral = integrate.quad(llc, 0, 1)
    assert integral[0] == pytest.approx(7, rel=1e-12)
