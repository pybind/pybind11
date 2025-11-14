from __future__ import annotations

import pytest

from pybind11_tests import numpy_scalars as m

np = pytest.importorskip("numpy")

NPY_SCALAR_TYPES = {
    np.bool_: False,
    np.int8: -7,
    np.int16: -15,
    np.int32: -31,
    np.int64: -63,
    np.uint8: 9,
    np.uint16: 17,
    np.uint32: 33,
    np.uint64: 65,
    np.single: 1.125,
    np.double: 1.25,
    np.complex64: 1 - 0.125j,
    np.complex128: 1 - 0.25j,
}

ALL_SCALAR_TYPES = tuple(NPY_SCALAR_TYPES.keys()) + (int, bool, float, bytes, str)


@pytest.mark.parametrize(
    ("npy_scalar_type", "expected_value"), NPY_SCALAR_TYPES.items()
)
def test_numpy_scalars(npy_scalar_type, expected_value):
    tpnm = npy_scalar_type.__name__.rstrip("_")
    test_tpnm = getattr(m, "test_" + tpnm)
    assert (
        test_tpnm.__doc__
        == f"test_{tpnm}(x: numpy.{tpnm}) -> tuple[str, numpy.{tpnm}]\n"
    )
    for tp in ALL_SCALAR_TYPES:
        value = tp(1)
        if tp is npy_scalar_type:
            result_tpnm, result_value = test_tpnm(value)
            assert result_tpnm == tpnm
            assert isinstance(result_value, npy_scalar_type)
            assert result_value == tp(expected_value)
        else:
            with pytest.raises(TypeError):
                test_tpnm(value)


def test_eq_ne():
    assert m.test_eq(np.int32(3), np.int32(3))
    assert not m.test_eq(np.int32(3), np.int32(5))
    assert not m.test_ne(np.int32(3), np.int32(3))
    assert m.test_ne(np.int32(3), np.int32(5))
