import sys
import pytest
from pybind11_tests import numpy_scalars as m

pytestmark = pytest.requires_numpy

SCALAR_TYPES = {}

with pytest.suppress(ImportError):
    import numpy as np

    SCALAR_TYPES = dict([
        (np.bool_, False),
        (np.int8, -7),
        (np.int16, -15),
        (np.int32, -31),
        (np.int64, -63),
        (np.uint8, 9),
        (np.uint16, 17),
        (np.uint32, 33),
        (np.uint64, 65),
        (np.single, 1.125),
        (np.double, 1.25),
        (np.longdouble, 1.5),
        (np.csingle, 1 - 0.125j),
        (np.cdouble, 1 - 0.25j),
        (np.clongdouble, 1 - 0.5j),
    ])

ALL_TYPES = [int, bool, float, bytes, str, type(None)] + list(SCALAR_TYPES)


def type_name(tp):
    try:
        if tp is np.longdouble:
            return 'longdouble'
        elif issubclass(tp, np.floating):
            return 'float' + str(8 * tp().itemsize)
        elif tp is np.clongdouble:
            return 'longcomplex'
        elif issubclass(tp, np.complexfloating):
            return 'complex' + str(8 * tp().itemsize)
        return tp.__name__.rstrip('_')
    except BaseException:
        # no numpy
        return str(tp)


@pytest.fixture(scope='module', params=list(SCALAR_TYPES), ids=type_name)
def scalar_type(request):
    return request.param


def expected_signature(tp):
    s = 'str' if sys.version_info[0] >= 3 else 'unicode'
    t = type_name(tp)
    return 'test_{t}(x: {t}) -> Tuple[{s}, {t}]\n'.format(s=s, t=t)


def test_numpy_scalars_single(scalar_type):
    expected = SCALAR_TYPES[scalar_type]
    name = type_name(scalar_type)
    func = getattr(m, 'test_' + name)
    assert func.__doc__ == expected_signature(scalar_type)
    for tp in ALL_TYPES:
        value = None if isinstance(None, tp) else tp(1)
        if tp is scalar_type:
            result = func(value)
            assert result[0] == name
            assert isinstance(result[1], tp)
            assert result[1] == tp(expected)
        else:
            with pytest.raises(TypeError):
                func(value)


def test_numpy_scalars_overload():
    func = m.test_numpy_scalars
    for tp in ALL_TYPES:
        value = None if isinstance(None, tp) else tp(1)
        if tp in SCALAR_TYPES:
            result = func(value)
            assert result[0] == type_name(tp)
            assert isinstance(result[1], tp)
            assert result[1] == tp(SCALAR_TYPES[tp])
        else:
            with pytest.raises(TypeError):
                func(value)
