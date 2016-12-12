import pytest

with pytest.suppress(ImportError):
    import numpy as np

    ref = np.array([[ 0,  3,  0,  0,  0, 11],
                    [22,  0,  0,  0, 17, 11],
                    [ 7,  5,  0,  1,  0, 11],
                    [ 0,  0,  0,  0,  0, 11],
                    [ 0,  0, 14,  0,  8, 11]])


def assert_equal_ref(mat):
    np.testing.assert_array_equal(mat, ref)


def assert_sparse_equal_ref(sparse_mat):
    assert_equal_ref(sparse_mat.todense())


@pytest.requires_eigen_and_numpy
def test_fixed():
    from pybind11_tests import fixed_r, fixed_c, fixed_passthrough_r, fixed_passthrough_c

    assert_equal_ref(fixed_c())
    assert_equal_ref(fixed_r())
    assert_equal_ref(fixed_passthrough_r(fixed_r()))
    assert_equal_ref(fixed_passthrough_c(fixed_c()))
    assert_equal_ref(fixed_passthrough_r(fixed_c()))
    assert_equal_ref(fixed_passthrough_c(fixed_r()))


@pytest.requires_eigen_and_numpy
def test_dense():
    from pybind11_tests import dense_r, dense_c, dense_passthrough_r, dense_passthrough_c

    assert_equal_ref(dense_r())
    assert_equal_ref(dense_c())
    assert_equal_ref(dense_passthrough_r(dense_r()))
    assert_equal_ref(dense_passthrough_c(dense_c()))
    assert_equal_ref(dense_passthrough_r(dense_c()))
    assert_equal_ref(dense_passthrough_c(dense_r()))


@pytest.requires_eigen_and_numpy
def test_nonunit_stride_from_python():
    from pybind11_tests import double_row, double_col, double_mat_cm, double_mat_rm

    counting_mat = np.arange(9.0, dtype=np.float32).reshape((3, 3))
    first_row = counting_mat[0, :]
    first_col = counting_mat[:, 0]
    assert np.array_equal(double_row(first_row), 2.0 * first_row)
    assert np.array_equal(double_col(first_row), 2.0 * first_row)
    assert np.array_equal(double_row(first_col), 2.0 * first_col)
    assert np.array_equal(double_col(first_col), 2.0 * first_col)

    counting_3d = np.arange(27.0, dtype=np.float32).reshape((3, 3, 3))
    slices = [counting_3d[0, :, :], counting_3d[:, 0, :], counting_3d[:, :, 0]]
    for slice_idx, ref_mat in enumerate(slices):
        assert np.array_equal(double_mat_cm(ref_mat), 2.0 * ref_mat)
        assert np.array_equal(double_mat_rm(ref_mat), 2.0 * ref_mat)


@pytest.requires_eigen_and_numpy
def test_nonunit_stride_to_python():
    from pybind11_tests import diagonal, diagonal_1, diagonal_n, block

    assert np.all(diagonal(ref) == ref.diagonal())
    assert np.all(diagonal_1(ref) == ref.diagonal(1))
    for i in range(-5, 7):
        assert np.all(diagonal_n(ref, i) == ref.diagonal(i)), "diagonal_n({})".format(i)

    assert np.all(block(ref, 2, 1, 3, 3) == ref[2:5, 1:4])
    assert np.all(block(ref, 1, 4, 4, 2) == ref[1:, 4:])
    assert np.all(block(ref, 1, 4, 3, 2) == ref[1:4, 4:])


@pytest.requires_eigen_and_numpy
def test_eigen_ref_to_python():
    from pybind11_tests import cholesky1, cholesky2, cholesky3, cholesky4, cholesky5, cholesky6

    chols = [cholesky1, cholesky2, cholesky3, cholesky4, cholesky5, cholesky6]
    for i, chol in enumerate(chols, start=1):
        mymat = chol(np.array([[1, 2, 4], [2, 13, 23], [4, 23, 77]]))
        assert np.all(mymat == np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])), "cholesky{}".format(i)


@pytest.requires_eigen_and_numpy
def test_special_matrix_objects():
    from pybind11_tests import incr_diag, symmetric_upper, symmetric_lower

    assert np.all(incr_diag(7) == np.diag([1, 2, 3, 4, 5, 6, 7]))

    asymm = np.array([[ 1,  2,  3,  4],
                      [ 5,  6,  7,  8],
                      [ 9, 10, 11, 12],
                      [13, 14, 15, 16]])
    symm_lower = np.array(asymm)
    symm_upper = np.array(asymm)
    for i in range(4):
        for j in range(i + 1, 4):
            symm_lower[i, j] = symm_lower[j, i]
            symm_upper[j, i] = symm_upper[i, j]

    assert np.all(symmetric_lower(asymm) == symm_lower)
    assert np.all(symmetric_upper(asymm) == symm_upper)


@pytest.requires_eigen_and_numpy
def test_dense_signature(doc):
    from pybind11_tests import double_col, double_row, double_mat_rm

    assert doc(double_col) == """
        double_col(arg0: numpy.ndarray[float32[m, 1]]) -> numpy.ndarray[float32[m, 1]]
    """
    assert doc(double_row) == """
        double_row(arg0: numpy.ndarray[float32[1, n]]) -> numpy.ndarray[float32[1, n]]
    """
    assert doc(double_mat_rm) == """
        double_mat_rm(arg0: numpy.ndarray[float32[m, n]]) -> numpy.ndarray[float32[m, n]]
    """


@pytest.requires_eigen_and_scipy
def test_sparse():
    from pybind11_tests import sparse_r, sparse_c, sparse_passthrough_r, sparse_passthrough_c

    assert_sparse_equal_ref(sparse_r())
    assert_sparse_equal_ref(sparse_c())
    assert_sparse_equal_ref(sparse_passthrough_r(sparse_r()))
    assert_sparse_equal_ref(sparse_passthrough_c(sparse_c()))
    assert_sparse_equal_ref(sparse_passthrough_r(sparse_c()))
    assert_sparse_equal_ref(sparse_passthrough_c(sparse_r()))


@pytest.requires_eigen_and_scipy
def test_sparse_signature(doc):
    from pybind11_tests import sparse_passthrough_r, sparse_passthrough_c

    assert doc(sparse_passthrough_r) == """
        sparse_passthrough_r(arg0: scipy.sparse.csr_matrix[float32]) -> scipy.sparse.csr_matrix[float32]
    """  # noqa: E501 line too long
    assert doc(sparse_passthrough_c) == """
        sparse_passthrough_c(arg0: scipy.sparse.csc_matrix[float32]) -> scipy.sparse.csc_matrix[float32]
    """  # noqa: E501 line too long
