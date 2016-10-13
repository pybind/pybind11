import pytest
import gc

with pytest.suppress(ImportError):
    import numpy as np


@pytest.fixture(scope='function')
def arr():
    return np.array([[1, 2, 3], [4, 5, 6]], '<u2')


@pytest.requires_numpy
def test_array_attributes():
    from pybind11_tests.array import (
        ndim, shape, strides, writeable, size, itemsize, nbytes, owndata
    )

    a = np.array(0, 'f8')
    assert ndim(a) == 0
    assert all(shape(a) == [])
    assert all(strides(a) == [])
    with pytest.raises(IndexError) as excinfo:
        shape(a, 0)
    assert str(excinfo.value) == 'invalid axis: 0 (ndim = 0)'
    with pytest.raises(IndexError) as excinfo:
        strides(a, 0)
    assert str(excinfo.value) == 'invalid axis: 0 (ndim = 0)'
    assert writeable(a)
    assert size(a) == 1
    assert itemsize(a) == 8
    assert nbytes(a) == 8
    assert owndata(a)

    a = np.array([[1, 2, 3], [4, 5, 6]], 'u2').view()
    a.flags.writeable = False
    assert ndim(a) == 2
    assert all(shape(a) == [2, 3])
    assert shape(a, 0) == 2
    assert shape(a, 1) == 3
    assert all(strides(a) == [6, 2])
    assert strides(a, 0) == 6
    assert strides(a, 1) == 2
    with pytest.raises(IndexError) as excinfo:
        shape(a, 2)
    assert str(excinfo.value) == 'invalid axis: 2 (ndim = 2)'
    with pytest.raises(IndexError) as excinfo:
        strides(a, 2)
    assert str(excinfo.value) == 'invalid axis: 2 (ndim = 2)'
    assert not writeable(a)
    assert size(a) == 6
    assert itemsize(a) == 2
    assert nbytes(a) == 12
    assert not owndata(a)


@pytest.requires_numpy
@pytest.mark.parametrize('args, ret', [([], 0), ([0], 0), ([1], 3), ([0, 1], 1), ([1, 2], 5)])
def test_index_offset(arr, args, ret):
    from pybind11_tests.array import index_at, index_at_t, offset_at, offset_at_t
    assert index_at(arr, *args) == ret
    assert index_at_t(arr, *args) == ret
    assert offset_at(arr, *args) == ret * arr.dtype.itemsize
    assert offset_at_t(arr, *args) == ret * arr.dtype.itemsize


@pytest.requires_numpy
def test_dim_check_fail(arr):
    from pybind11_tests.array import (index_at, index_at_t, offset_at, offset_at_t, data, data_t,
                                      mutate_data, mutate_data_t)
    for func in (index_at, index_at_t, offset_at, offset_at_t, data, data_t,
                 mutate_data, mutate_data_t):
        with pytest.raises(IndexError) as excinfo:
            func(arr, 1, 2, 3)
        assert str(excinfo.value) == 'too many indices for an array: 3 (ndim = 2)'


@pytest.requires_numpy
@pytest.mark.parametrize('args, ret',
                         [([], [1, 2, 3, 4, 5, 6]),
                          ([1], [4, 5, 6]),
                          ([0, 1], [2, 3, 4, 5, 6]),
                          ([1, 2], [6])])
def test_data(arr, args, ret):
    from pybind11_tests.array import data, data_t
    assert all(data_t(arr, *args) == ret)
    assert all(data(arr, *args)[::2] == ret)
    assert all(data(arr, *args)[1::2] == 0)


@pytest.requires_numpy
def test_mutate_readonly(arr):
    from pybind11_tests.array import mutate_data, mutate_data_t, mutate_at_t
    arr.flags.writeable = False
    for func, args in (mutate_data, ()), (mutate_data_t, ()), (mutate_at_t, (0, 0)):
        with pytest.raises(RuntimeError) as excinfo:
            func(arr, *args)
        assert str(excinfo.value) == 'array is not writeable'


@pytest.requires_numpy
@pytest.mark.parametrize('dim', [0, 1, 3])
def test_at_fail(arr, dim):
    from pybind11_tests.array import at_t, mutate_at_t
    for func in at_t, mutate_at_t:
        with pytest.raises(IndexError) as excinfo:
            func(arr, *([0] * dim))
        assert str(excinfo.value) == 'index dimension mismatch: {} (ndim = 2)'.format(dim)


@pytest.requires_numpy
def test_at(arr):
    from pybind11_tests.array import at_t, mutate_at_t

    assert at_t(arr, 0, 2) == 3
    assert at_t(arr, 1, 0) == 4

    assert all(mutate_at_t(arr, 0, 2).ravel() == [1, 2, 4, 4, 5, 6])
    assert all(mutate_at_t(arr, 1, 0).ravel() == [1, 2, 4, 5, 5, 6])


@pytest.requires_numpy
def test_mutate_data(arr):
    from pybind11_tests.array import mutate_data, mutate_data_t

    assert all(mutate_data(arr).ravel() == [2, 4, 6, 8, 10, 12])
    assert all(mutate_data(arr).ravel() == [4, 8, 12, 16, 20, 24])
    assert all(mutate_data(arr, 1).ravel() == [4, 8, 12, 32, 40, 48])
    assert all(mutate_data(arr, 0, 1).ravel() == [4, 16, 24, 64, 80, 96])
    assert all(mutate_data(arr, 1, 2).ravel() == [4, 16, 24, 64, 80, 192])

    assert all(mutate_data_t(arr).ravel() == [5, 17, 25, 65, 81, 193])
    assert all(mutate_data_t(arr).ravel() == [6, 18, 26, 66, 82, 194])
    assert all(mutate_data_t(arr, 1).ravel() == [6, 18, 26, 67, 83, 195])
    assert all(mutate_data_t(arr, 0, 1).ravel() == [6, 19, 27, 68, 84, 196])
    assert all(mutate_data_t(arr, 1, 2).ravel() == [6, 19, 27, 68, 84, 197])


@pytest.requires_numpy
def test_bounds_check(arr):
    from pybind11_tests.array import (index_at, index_at_t, data, data_t,
                                      mutate_data, mutate_data_t, at_t, mutate_at_t)
    funcs = (index_at, index_at_t, data, data_t,
             mutate_data, mutate_data_t, at_t, mutate_at_t)
    for func in funcs:
        with pytest.raises(IndexError) as excinfo:
            index_at(arr, 2, 0)
        assert str(excinfo.value) == 'index 2 is out of bounds for axis 0 with size 2'
        with pytest.raises(IndexError) as excinfo:
            index_at(arr, 0, 4)
        assert str(excinfo.value) == 'index 4 is out of bounds for axis 1 with size 3'


@pytest.requires_numpy
def test_make_c_f_array():
    from pybind11_tests.array import (
        make_c_array, make_f_array
    )
    assert make_c_array().flags.c_contiguous
    assert not make_c_array().flags.f_contiguous
    assert make_f_array().flags.f_contiguous
    assert not make_f_array().flags.c_contiguous


@pytest.requires_numpy
def test_wrap():
    from pybind11_tests.array import wrap

    def assert_references(A, B):
        assert A is not B
        assert A.__array_interface__['data'][0] == \
               B.__array_interface__['data'][0]
        assert A.shape == B.shape
        assert A.strides == B.strides
        assert A.flags.c_contiguous == B.flags.c_contiguous
        assert A.flags.f_contiguous == B.flags.f_contiguous
        assert A.flags.writeable == B.flags.writeable
        assert A.flags.aligned == B.flags.aligned
        assert A.flags.updateifcopy == B.flags.updateifcopy
        assert np.all(A == B)
        assert not B.flags.owndata
        assert B.base is A
        if A.flags.writeable and A.ndim == 2:
            A[0, 0] = 1234
            assert B[0, 0] == 1234

    A1 = np.array([1, 2], dtype=np.int16)
    assert A1.flags.owndata and A1.base is None
    A2 = wrap(A1)
    assert_references(A1, A2)

    A1 = np.array([[1, 2], [3, 4]], dtype=np.float32, order='F')
    assert A1.flags.owndata and A1.base is None
    A2 = wrap(A1)
    assert_references(A1, A2)

    A1 = np.array([[1, 2], [3, 4]], dtype=np.float32, order='C')
    A1.flags.writeable = False
    A2 = wrap(A1)
    assert_references(A1, A2)

    A1 = np.random.random((4, 4, 4))
    A2 = wrap(A1)
    assert_references(A1, A2)

    A1 = A1.transpose()
    A2 = wrap(A1)
    assert_references(A1, A2)

    A1 = A1.diagonal()
    A2 = wrap(A1)
    assert_references(A1, A2)


@pytest.requires_numpy
def test_numpy_view(capture):
    from pybind11_tests.array import ArrayClass
    with capture:
        ac = ArrayClass()
        ac_view_1 = ac.numpy_view()
        ac_view_2 = ac.numpy_view()
        assert np.all(ac_view_1 == np.array([1, 2], dtype=np.int32))
        del ac
        gc.collect()
    assert capture == """
        ArrayClass()
        ArrayClass::numpy_view()
        ArrayClass::numpy_view()
    """
    ac_view_1[0] = 4
    ac_view_1[1] = 3
    assert ac_view_2[0] == 4
    assert ac_view_2[1] == 3
    with capture:
        del ac_view_1
        del ac_view_2
        gc.collect()
    assert capture == """
        ~ArrayClass()
    """
