import sys

import pytest

from pybind11_tests import numpy_scalars as m

np = pytest.importorskip("numpy")

SCALAR_TYPES = {
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
ALL_TYPES = [int, bool, float, bytes, str] + list(SCALAR_TYPES)


def type_name(tp):
    try:
        return tp.__name__.rstrip("_")
    except BaseException:
        # no numpy
        return str(tp)


@pytest.fixture(scope="module", params=list(SCALAR_TYPES), ids=type_name)
def scalar_type(request):
    return request.param


def expected_signature(tp):
    s = "str" if sys.version_info[0] >= 3 else "unicode"
    t = type_name(tp)
    return f"test_{t}(x: {t}) -> tuple[{s}, {t}]\n"


def test_numpy_scalars(scalar_type):
    expected = SCALAR_TYPES[scalar_type]
    name = type_name(scalar_type)
    func = getattr(m, "test_" + name)
    assert func.__doc__ == expected_signature(scalar_type)
    for tp in ALL_TYPES:
        value = tp(1)
        if tp is scalar_type:
            result = func(value)
            assert result[0] == name
            assert isinstance(result[1], tp)
            assert result[1] == tp(expected)
        else:
            with pytest.raises(TypeError):
                func(value)
