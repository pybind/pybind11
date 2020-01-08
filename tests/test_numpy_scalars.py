import sys
import pytest
from pybind11_tests import numpy_scalars as m

pytestmark = pytest.requires_numpy

with pytest.suppress(ImportError):
    import numpy as np


@pytest.fixture(scope='module')
def scalar_types():
    return [
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
    ]


@pytest.fixture(scope='module')
def other_types():
    return [int, bool, float, bytes, str, np.complex64, type(None)]


def signature(tp):
    name = tp.__name__.rstrip('_')
    return 'test_numpy_scalars(x: {tp}) -> Tuple[str, {tp}]'.format(tp=name)


def test_numpy_scalars_single(scalar_types, other_types):
    s_tp = 'str' if sys.version_info[0] >= 3 else 'unicode'
    for scalar_type in scalar_types:
        name = scalar_type.__name__.rstrip('_')
        func = getattr(m, 'test_' + name)
        sig = 'test_{tp}(x: {tp}) -> Tuple[{s_tp}, {tp}]\n'.format(tp=name, s_tp=s_tp)
        assert func.__doc__ == sig
        for tp in (scalar_types + other_types):
            value = None if isinstance(None, tp) else tp()
            if tp is scalar_type:
                result = func(value)
                assert result[0] == name
                assert isinstance(result[1], tp)
                assert result[1] == tp(1)
            else:
                with pytest.raises(TypeError):
                    func(value)


def test_numpy_scalars_overload(scalar_types, other_types):
    func = m.test_numpy_scalars
    for tp in (scalar_types + other_types):
        value = None if isinstance(None, tp) else tp()
        if tp in scalar_types:
            name = tp.__name__.rstrip('_')
            result = func(value)
            assert result[0] == name
            assert isinstance(result[1], tp)
            assert result[1] == tp(1)
        else:
            with pytest.raises(TypeError):
                func(value)
