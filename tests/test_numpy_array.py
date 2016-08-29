import pytest

with pytest.suppress(ImportError):
    import numpy as np


@pytest.requires_numpy
def test_array_attributes():
    from pybind11_tests import (get_arr_ndim, get_arr_shape, get_arr_strides, get_arr_writeable,
                                get_arr_size, get_arr_itemsize, get_arr_nbytes, get_arr_owndata)

    a = np.array(0, 'f8')
    assert get_arr_ndim(a) == 0
    assert get_arr_shape(a) == []
    assert get_arr_strides(a) == []
    with pytest.raises(RuntimeError):
        get_arr_shape(a, 1)
    with pytest.raises(RuntimeError):
        get_arr_strides(a, 0)
    assert get_arr_writeable(a)
    assert get_arr_size(a) == 1
    assert get_arr_itemsize(a) == 8
    assert get_arr_nbytes(a) == 8
    assert get_arr_owndata(a)

    a = np.array([[1, 2, 3], [4, 5, 6]], 'u2').view()
    a.flags.writeable = False
    assert get_arr_ndim(a) == 2
    assert get_arr_shape(a) == [2, 3]
    assert get_arr_shape(a, 0) == 2
    assert get_arr_shape(a, 1) == 3
    assert get_arr_strides(a) == [6, 2]
    assert get_arr_strides(a, 0) == 6
    assert get_arr_strides(a, 1) == 2
    with pytest.raises(RuntimeError):
        get_arr_shape(a, 2)
    with pytest.raises(RuntimeError):
        get_arr_strides(a, 2)
    assert not get_arr_writeable(a)
    assert get_arr_size(a) == 6
    assert get_arr_itemsize(a) == 2
    assert get_arr_nbytes(a) == 12
    assert not get_arr_owndata(a)
