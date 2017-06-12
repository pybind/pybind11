import pytest

pytestmark = pytest.requires_eigen_and_numpy

with pytest.suppress(ImportError):
    import numpy as np

    ref = np.array([[ 0.,  3,  0,  0,  0, 11],
                    [22,  0,  0,  0, 17, 11],
                    [ 7,  5,  0,  1,  0, 11],
                    [ 0,  0,  0,  0,  0, 11],
                    [ 0,  0, 14,  0,  8, 11]])


def assert_equal_ref(mat):
    np.testing.assert_array_equal(mat, ref)


def assert_sparse_equal_ref(sparse_mat):
    assert_equal_ref(sparse_mat.todense())


def test_fixed():
    from pybind11_tests import fixed_r, fixed_c, fixed_copy_r, fixed_copy_c

    assert_equal_ref(fixed_c())
    assert_equal_ref(fixed_r())
    assert_equal_ref(fixed_copy_r(fixed_r()))
    assert_equal_ref(fixed_copy_c(fixed_c()))
    assert_equal_ref(fixed_copy_r(fixed_c()))
    assert_equal_ref(fixed_copy_c(fixed_r()))


def test_dense():
    from pybind11_tests import dense_r, dense_c, dense_copy_r, dense_copy_c

    assert_equal_ref(dense_r())
    assert_equal_ref(dense_c())
    assert_equal_ref(dense_copy_r(dense_r()))
    assert_equal_ref(dense_copy_c(dense_c()))
    assert_equal_ref(dense_copy_r(dense_c()))
    assert_equal_ref(dense_copy_c(dense_r()))


def test_partially_fixed():
    from pybind11_tests import (partial_copy_four_rm_r, partial_copy_four_rm_c,
                                partial_copy_four_cm_r, partial_copy_four_cm_c)

    ref2 = np.array([[0., 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
    np.testing.assert_array_equal(partial_copy_four_rm_r(ref2), ref2)
    np.testing.assert_array_equal(partial_copy_four_rm_c(ref2), ref2)
    np.testing.assert_array_equal(partial_copy_four_rm_r(ref2[:, 1]), ref2[:, [1]])
    np.testing.assert_array_equal(partial_copy_four_rm_c(ref2[0, :]), ref2[[0], :])
    np.testing.assert_array_equal(partial_copy_four_rm_r(ref2[:, (0, 2)]), ref2[:, (0, 2)])
    np.testing.assert_array_equal(
        partial_copy_four_rm_c(ref2[(3, 1, 2), :]), ref2[(3, 1, 2), :])

    np.testing.assert_array_equal(partial_copy_four_cm_r(ref2), ref2)
    np.testing.assert_array_equal(partial_copy_four_cm_c(ref2), ref2)
    np.testing.assert_array_equal(partial_copy_four_cm_r(ref2[:, 1]), ref2[:, [1]])
    np.testing.assert_array_equal(partial_copy_four_cm_c(ref2[0, :]), ref2[[0], :])
    np.testing.assert_array_equal(partial_copy_four_cm_r(ref2[:, (0, 2)]), ref2[:, (0, 2)])
    np.testing.assert_array_equal(
        partial_copy_four_cm_c(ref2[(3, 1, 2), :]), ref2[(3, 1, 2), :])

    # TypeError should be raise for a shape mismatch
    functions = [partial_copy_four_rm_r, partial_copy_four_rm_c,
                 partial_copy_four_cm_r, partial_copy_four_cm_c]
    matrix_with_wrong_shape = [[1, 2],
                               [3, 4]]
    for f in functions:
        with pytest.raises(TypeError) as excinfo:
            f(matrix_with_wrong_shape)
        assert "incompatible function arguments" in str(excinfo.value)


def test_mutator_descriptors():
    from pybind11_tests import fixed_mutator_r, fixed_mutator_c, fixed_mutator_a
    zr = np.arange(30, dtype='float32').reshape(5, 6)  # row-major
    zc = zr.reshape(6, 5).transpose()  # column-major

    fixed_mutator_r(zr)
    fixed_mutator_c(zc)
    fixed_mutator_a(zr)
    fixed_mutator_a(zc)
    with pytest.raises(TypeError) as excinfo:
        fixed_mutator_r(zc)
    assert ('(arg0: numpy.ndarray[float32[5, 6], flags.writeable, flags.c_contiguous]) -> None'
            in str(excinfo.value))
    with pytest.raises(TypeError) as excinfo:
        fixed_mutator_c(zr)
    assert ('(arg0: numpy.ndarray[float32[5, 6], flags.writeable, flags.f_contiguous]) -> None'
            in str(excinfo.value))
    with pytest.raises(TypeError) as excinfo:
        fixed_mutator_a(np.array([[1, 2], [3, 4]], dtype='float32'))
    assert ('(arg0: numpy.ndarray[float32[5, 6], flags.writeable]) -> None'
            in str(excinfo.value))
    zr.flags.writeable = False
    with pytest.raises(TypeError):
        fixed_mutator_r(zr)
    with pytest.raises(TypeError):
        fixed_mutator_a(zr)


def test_cpp_casting():
    from pybind11_tests import (cpp_copy, cpp_ref_c, cpp_ref_r, cpp_ref_any,
                                fixed_r, fixed_c, get_cm_ref, get_rm_ref, ReturnTester)
    assert cpp_copy(fixed_r()) == 22.
    assert cpp_copy(fixed_c()) == 22.
    z = np.array([[5., 6], [7, 8]])
    assert cpp_copy(z) == 7.
    assert cpp_copy(get_cm_ref()) == 21.
    assert cpp_copy(get_rm_ref()) == 21.
    assert cpp_ref_c(get_cm_ref()) == 21.
    assert cpp_ref_r(get_rm_ref()) == 21.
    with pytest.raises(RuntimeError) as excinfo:
        # Can't reference fixed_c: it contains floats, cpp_ref_any wants doubles
        cpp_ref_any(fixed_c())
    assert 'Unable to cast Python instance' in str(excinfo.value)
    with pytest.raises(RuntimeError) as excinfo:
        # Can't reference fixed_r: it contains floats, cpp_ref_any wants doubles
        cpp_ref_any(fixed_r())
    assert 'Unable to cast Python instance' in str(excinfo.value)
    assert cpp_ref_any(ReturnTester.create()) == 1.

    assert cpp_ref_any(get_cm_ref()) == 21.
    assert cpp_ref_any(get_cm_ref()) == 21.


def test_pass_readonly_array():
    from pybind11_tests import fixed_copy_r, fixed_r, fixed_r_const
    z = np.full((5, 6), 42.0)
    z.flags.writeable = False
    np.testing.assert_array_equal(z, fixed_copy_r(z))
    np.testing.assert_array_equal(fixed_r_const(), fixed_r())
    assert not fixed_r_const().flags.writeable
    np.testing.assert_array_equal(fixed_copy_r(fixed_r_const()), fixed_r_const())


def test_nonunit_stride_from_python():
    from pybind11_tests import (
        double_row, double_col, double_complex, double_mat_cm, double_mat_rm,
        double_threec, double_threer)

    counting_mat = np.arange(9.0, dtype=np.float32).reshape((3, 3))
    second_row = counting_mat[1, :]
    second_col = counting_mat[:, 1]
    np.testing.assert_array_equal(double_row(second_row), 2.0 * second_row)
    np.testing.assert_array_equal(double_col(second_row), 2.0 * second_row)
    np.testing.assert_array_equal(double_complex(second_row), 2.0 * second_row)
    np.testing.assert_array_equal(double_row(second_col), 2.0 * second_col)
    np.testing.assert_array_equal(double_col(second_col), 2.0 * second_col)
    np.testing.assert_array_equal(double_complex(second_col), 2.0 * second_col)

    counting_3d = np.arange(27.0, dtype=np.float32).reshape((3, 3, 3))
    slices = [counting_3d[0, :, :], counting_3d[:, 0, :], counting_3d[:, :, 0]]
    for slice_idx, ref_mat in enumerate(slices):
        np.testing.assert_array_equal(double_mat_cm(ref_mat), 2.0 * ref_mat)
        np.testing.assert_array_equal(double_mat_rm(ref_mat), 2.0 * ref_mat)

    # Mutator:
    double_threer(second_row)
    double_threec(second_col)
    np.testing.assert_array_equal(counting_mat, [[0., 2, 2], [6, 16, 10], [6, 14, 8]])


def test_negative_stride_from_python(msg):
    from pybind11_tests import (
        double_row, double_col, double_complex, double_mat_cm, double_mat_rm,
        double_threec, double_threer)

    # Eigen doesn't support (as of yet) negative strides. When a function takes an Eigen
    # matrix by copy or const reference, we can pass a numpy array that has negative strides.
    # Otherwise, an exception will be thrown as Eigen will not be able to map the numpy array.

    counting_mat = np.arange(9.0, dtype=np.float32).reshape((3, 3))
    counting_mat = counting_mat[::-1, ::-1]
    second_row = counting_mat[1, :]
    second_col = counting_mat[:, 1]
    np.testing.assert_array_equal(double_row(second_row), 2.0 * second_row)
    np.testing.assert_array_equal(double_col(second_row), 2.0 * second_row)
    np.testing.assert_array_equal(double_complex(second_row), 2.0 * second_row)
    np.testing.assert_array_equal(double_row(second_col), 2.0 * second_col)
    np.testing.assert_array_equal(double_col(second_col), 2.0 * second_col)
    np.testing.assert_array_equal(double_complex(second_col), 2.0 * second_col)

    counting_3d = np.arange(27.0, dtype=np.float32).reshape((3, 3, 3))
    counting_3d = counting_3d[::-1, ::-1, ::-1]
    slices = [counting_3d[0, :, :], counting_3d[:, 0, :], counting_3d[:, :, 0]]
    for slice_idx, ref_mat in enumerate(slices):
        np.testing.assert_array_equal(double_mat_cm(ref_mat), 2.0 * ref_mat)
        np.testing.assert_array_equal(double_mat_rm(ref_mat), 2.0 * ref_mat)

    # Mutator:
    with pytest.raises(TypeError) as excinfo:
        double_threer(second_row)
    assert msg(excinfo.value) == """
    double_threer(): incompatible function arguments. The following argument types are supported:
        1. (arg0: numpy.ndarray[float32[1, 3], flags.writeable]) -> None

    Invoked with: array([ 5.,  4.,  3.], dtype=float32)
"""

    with pytest.raises(TypeError) as excinfo:
        double_threec(second_col)
    assert msg(excinfo.value) == """
    double_threec(): incompatible function arguments. The following argument types are supported:
        1. (arg0: numpy.ndarray[float32[3, 1], flags.writeable]) -> None

    Invoked with: array([ 7.,  4.,  1.], dtype=float32)
"""


def test_nonunit_stride_to_python():
    from pybind11_tests import diagonal, diagonal_1, diagonal_n, block

    assert np.all(diagonal(ref) == ref.diagonal())
    assert np.all(diagonal_1(ref) == ref.diagonal(1))
    for i in range(-5, 7):
        assert np.all(diagonal_n(ref, i) == ref.diagonal(i)), "diagonal_n({})".format(i)

    assert np.all(block(ref, 2, 1, 3, 3) == ref[2:5, 1:4])
    assert np.all(block(ref, 1, 4, 4, 2) == ref[1:, 4:])
    assert np.all(block(ref, 1, 4, 3, 2) == ref[1:4, 4:])


def test_eigen_ref_to_python():
    from pybind11_tests import cholesky1, cholesky2, cholesky3, cholesky4

    chols = [cholesky1, cholesky2, cholesky3, cholesky4]
    for i, chol in enumerate(chols, start=1):
        mymat = chol(np.array([[1., 2, 4], [2, 13, 23], [4, 23, 77]]))
        assert np.all(mymat == np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])), "cholesky{}".format(i)


def assign_both(a1, a2, r, c, v):
    a1[r, c] = v
    a2[r, c] = v


def array_copy_but_one(a, r, c, v):
    z = np.array(a, copy=True)
    z[r, c] = v
    return z


def test_eigen_return_references():
    """Tests various ways of returning references and non-referencing copies"""
    from pybind11_tests import ReturnTester
    master = np.ones((10, 10))
    a = ReturnTester()
    a_get1 = a.get()
    assert not a_get1.flags.owndata and a_get1.flags.writeable
    assign_both(a_get1, master, 3, 3, 5)
    a_get2 = a.get_ptr()
    assert not a_get2.flags.owndata and a_get2.flags.writeable
    assign_both(a_get1, master, 2, 3, 6)

    a_view1 = a.view()
    assert not a_view1.flags.owndata and not a_view1.flags.writeable
    with pytest.raises(ValueError):
        a_view1[2, 3] = 4
    a_view2 = a.view_ptr()
    assert not a_view2.flags.owndata and not a_view2.flags.writeable
    with pytest.raises(ValueError):
        a_view2[2, 3] = 4

    a_copy1 = a.copy_get()
    assert a_copy1.flags.owndata and a_copy1.flags.writeable
    np.testing.assert_array_equal(a_copy1, master)
    a_copy1[7, 7] = -44  # Shouldn't affect anything else
    c1want = array_copy_but_one(master, 7, 7, -44)
    a_copy2 = a.copy_view()
    assert a_copy2.flags.owndata and a_copy2.flags.writeable
    np.testing.assert_array_equal(a_copy2, master)
    a_copy2[4, 4] = -22  # Shouldn't affect anything else
    c2want = array_copy_but_one(master, 4, 4, -22)

    a_ref1 = a.ref()
    assert not a_ref1.flags.owndata and a_ref1.flags.writeable
    assign_both(a_ref1, master, 1, 1, 15)
    a_ref2 = a.ref_const()
    assert not a_ref2.flags.owndata and not a_ref2.flags.writeable
    with pytest.raises(ValueError):
        a_ref2[5, 5] = 33
    a_ref3 = a.ref_safe()
    assert not a_ref3.flags.owndata and a_ref3.flags.writeable
    assign_both(a_ref3, master, 0, 7, 99)
    a_ref4 = a.ref_const_safe()
    assert not a_ref4.flags.owndata and not a_ref4.flags.writeable
    with pytest.raises(ValueError):
        a_ref4[7, 0] = 987654321

    a_copy3 = a.copy_ref()
    assert a_copy3.flags.owndata and a_copy3.flags.writeable
    np.testing.assert_array_equal(a_copy3, master)
    a_copy3[8, 1] = 11
    c3want = array_copy_but_one(master, 8, 1, 11)
    a_copy4 = a.copy_ref_const()
    assert a_copy4.flags.owndata and a_copy4.flags.writeable
    np.testing.assert_array_equal(a_copy4, master)
    a_copy4[8, 4] = 88
    c4want = array_copy_but_one(master, 8, 4, 88)

    a_block1 = a.block(3, 3, 2, 2)
    assert not a_block1.flags.owndata and a_block1.flags.writeable
    a_block1[0, 0] = 55
    master[3, 3] = 55
    a_block2 = a.block_safe(2, 2, 3, 2)
    assert not a_block2.flags.owndata and a_block2.flags.writeable
    a_block2[2, 1] = -123
    master[4, 3] = -123
    a_block3 = a.block_const(6, 7, 4, 3)
    assert not a_block3.flags.owndata and not a_block3.flags.writeable
    with pytest.raises(ValueError):
        a_block3[2, 2] = -44444

    a_copy5 = a.copy_block(2, 2, 2, 3)
    assert a_copy5.flags.owndata and a_copy5.flags.writeable
    np.testing.assert_array_equal(a_copy5, master[2:4, 2:5])
    a_copy5[1, 1] = 777
    c5want = array_copy_but_one(master[2:4, 2:5], 1, 1, 777)

    a_corn1 = a.corners()
    assert not a_corn1.flags.owndata and a_corn1.flags.writeable
    a_corn1 *= 50
    a_corn1[1, 1] = 999
    master[0, 0] = 50
    master[0, 9] = 50
    master[9, 0] = 50
    master[9, 9] = 999
    a_corn2 = a.corners_const()
    assert not a_corn2.flags.owndata and not a_corn2.flags.writeable
    with pytest.raises(ValueError):
        a_corn2[1, 0] = 51

    # All of the changes made all the way along should be visible everywhere
    # now (except for the copies, of course)
    np.testing.assert_array_equal(a_get1, master)
    np.testing.assert_array_equal(a_get2, master)
    np.testing.assert_array_equal(a_view1, master)
    np.testing.assert_array_equal(a_view2, master)
    np.testing.assert_array_equal(a_ref1, master)
    np.testing.assert_array_equal(a_ref2, master)
    np.testing.assert_array_equal(a_ref3, master)
    np.testing.assert_array_equal(a_ref4, master)
    np.testing.assert_array_equal(a_block1, master[3:5, 3:5])
    np.testing.assert_array_equal(a_block2, master[2:5, 2:4])
    np.testing.assert_array_equal(a_block3, master[6:10, 7:10])
    np.testing.assert_array_equal(a_corn1, master[0::master.shape[0] - 1, 0::master.shape[1] - 1])
    np.testing.assert_array_equal(a_corn2, master[0::master.shape[0] - 1, 0::master.shape[1] - 1])

    np.testing.assert_array_equal(a_copy1, c1want)
    np.testing.assert_array_equal(a_copy2, c2want)
    np.testing.assert_array_equal(a_copy3, c3want)
    np.testing.assert_array_equal(a_copy4, c4want)
    np.testing.assert_array_equal(a_copy5, c5want)


def assert_keeps_alive(cl, method, *args):
    from pybind11_tests import ConstructorStats
    cstats = ConstructorStats.get(cl)
    start_with = cstats.alive()
    a = cl()
    assert cstats.alive() == start_with + 1
    z = method(a, *args)
    assert cstats.alive() == start_with + 1
    del a
    # Here's the keep alive in action:
    assert cstats.alive() == start_with + 1
    del z
    # Keep alive should have expired:
    assert cstats.alive() == start_with


def test_eigen_keepalive():
    from pybind11_tests import ReturnTester, ConstructorStats
    a = ReturnTester()

    cstats = ConstructorStats.get(ReturnTester)
    assert cstats.alive() == 1
    unsafe = [a.ref(), a.ref_const(), a.block(1, 2, 3, 4)]
    copies = [a.copy_get(), a.copy_view(), a.copy_ref(), a.copy_ref_const(),
              a.copy_block(4, 3, 2, 1)]
    del a
    assert cstats.alive() == 0
    del unsafe
    del copies

    for meth in [ReturnTester.get, ReturnTester.get_ptr, ReturnTester.view,
                 ReturnTester.view_ptr, ReturnTester.ref_safe, ReturnTester.ref_const_safe,
                 ReturnTester.corners, ReturnTester.corners_const]:
        assert_keeps_alive(ReturnTester, meth)

    for meth in [ReturnTester.block_safe, ReturnTester.block_const]:
        assert_keeps_alive(ReturnTester, meth, 4, 3, 2, 1)


def test_eigen_ref_mutators():
    """Tests whether Eigen can mutate numpy values"""
    from pybind11_tests import add_rm, add_cm, add_any, add1, add2
    orig = np.array([[1., 2, 3], [4, 5, 6], [7, 8, 9]])
    zr = np.array(orig)
    zc = np.array(orig, order='F')
    add_rm(zr, 1, 0, 100)
    assert np.all(zr == np.array([[1., 2, 3], [104, 5, 6], [7, 8, 9]]))
    add_cm(zc, 1, 0, 200)
    assert np.all(zc == np.array([[1., 2, 3], [204, 5, 6], [7, 8, 9]]))

    add_any(zr, 1, 0, 20)
    assert np.all(zr == np.array([[1., 2, 3], [124, 5, 6], [7, 8, 9]]))
    add_any(zc, 1, 0, 10)
    assert np.all(zc == np.array([[1., 2, 3], [214, 5, 6], [7, 8, 9]]))

    # Can't reference a col-major array with a row-major Ref, and vice versa:
    with pytest.raises(TypeError):
        add_rm(zc, 1, 0, 1)
    with pytest.raises(TypeError):
        add_cm(zr, 1, 0, 1)

    # Overloads:
    add1(zr, 1, 0, -100)
    add2(zr, 1, 0, -20)
    assert np.all(zr == orig)
    add1(zc, 1, 0, -200)
    add2(zc, 1, 0, -10)
    assert np.all(zc == orig)

    # a non-contiguous slice (this won't work on either the row- or
    # column-contiguous refs, but should work for the any)
    cornersr = zr[0::2, 0::2]
    cornersc = zc[0::2, 0::2]

    assert np.all(cornersr == np.array([[1., 3], [7, 9]]))
    assert np.all(cornersc == np.array([[1., 3], [7, 9]]))

    with pytest.raises(TypeError):
        add_rm(cornersr, 0, 1, 25)
    with pytest.raises(TypeError):
        add_cm(cornersr, 0, 1, 25)
    with pytest.raises(TypeError):
        add_rm(cornersc, 0, 1, 25)
    with pytest.raises(TypeError):
        add_cm(cornersc, 0, 1, 25)
    add_any(cornersr, 0, 1, 25)
    add_any(cornersc, 0, 1, 44)
    assert np.all(zr == np.array([[1., 2, 28], [4, 5, 6], [7, 8, 9]]))
    assert np.all(zc == np.array([[1., 2, 47], [4, 5, 6], [7, 8, 9]]))

    # You shouldn't be allowed to pass a non-writeable array to a mutating Eigen method:
    zro = zr[0:4, 0:4]
    zro.flags.writeable = False
    with pytest.raises(TypeError):
        add_rm(zro, 0, 0, 0)
    with pytest.raises(TypeError):
        add_any(zro, 0, 0, 0)
    with pytest.raises(TypeError):
        add1(zro, 0, 0, 0)
    with pytest.raises(TypeError):
        add2(zro, 0, 0, 0)

    # integer array shouldn't be passable to a double-matrix-accepting mutating func:
    zi = np.array([[1, 2], [3, 4]])
    with pytest.raises(TypeError):
        add_rm(zi)


def test_numpy_ref_mutators():
    """Tests numpy mutating Eigen matrices (for returned Eigen::Ref<...>s)"""
    from pybind11_tests import (
        get_cm_ref, get_cm_const_ref, get_rm_ref, get_rm_const_ref, reset_refs)
    reset_refs()  # In case another test already changed it

    zc = get_cm_ref()
    zcro = get_cm_const_ref()
    zr = get_rm_ref()
    zrro = get_rm_const_ref()

    assert [zc[1, 2], zcro[1, 2], zr[1, 2], zrro[1, 2]] == [23] * 4

    assert not zc.flags.owndata and zc.flags.writeable
    assert not zr.flags.owndata and zr.flags.writeable
    assert not zcro.flags.owndata and not zcro.flags.writeable
    assert not zrro.flags.owndata and not zrro.flags.writeable

    zc[1, 2] = 99
    expect = np.array([[11., 12, 13], [21, 22, 99], [31, 32, 33]])
    # We should have just changed zc, of course, but also zcro and the original eigen matrix
    assert np.all(zc == expect)
    assert np.all(zcro == expect)
    assert np.all(get_cm_ref() == expect)

    zr[1, 2] = 99
    assert np.all(zr == expect)
    assert np.all(zrro == expect)
    assert np.all(get_rm_ref() == expect)

    # Make sure the readonly ones are numpy-readonly:
    with pytest.raises(ValueError):
        zcro[1, 2] = 6
    with pytest.raises(ValueError):
        zrro[1, 2] = 6

    # We should be able to explicitly copy like this (and since we're copying,
    # the const should drop away)
    y1 = np.array(get_cm_const_ref())

    assert y1.flags.owndata and y1.flags.writeable
    # We should get copies of the eigen data, which was modified above:
    assert y1[1, 2] == 99
    y1[1, 2] += 12
    assert y1[1, 2] == 111
    assert zc[1, 2] == 99  # Make sure we aren't referencing the original


def test_both_ref_mutators():
    """Tests a complex chain of nested eigen/numpy references"""
    from pybind11_tests import (
        incr_matrix, get_cm_ref, incr_matrix_any, even_cols, even_rows, reset_refs)
    reset_refs()  # In case another test already changed it

    z = get_cm_ref()  # numpy -> eigen
    z[0, 2] -= 3
    z2 = incr_matrix(z, 1)  # numpy -> eigen -> numpy -> eigen
    z2[1, 1] += 6
    z3 = incr_matrix(z, 2)  # (numpy -> eigen)^3
    z3[2, 2] += -5
    z4 = incr_matrix(z, 3)  # (numpy -> eigen)^4
    z4[1, 1] -= 1
    z5 = incr_matrix(z, 4)  # (numpy -> eigen)^5
    z5[0, 0] = 0
    assert np.all(z == z2)
    assert np.all(z == z3)
    assert np.all(z == z4)
    assert np.all(z == z5)
    expect = np.array([[0., 22, 20], [31, 37, 33], [41, 42, 38]])
    assert np.all(z == expect)

    y = np.array(range(100), dtype='float64').reshape(10, 10)
    y2 = incr_matrix_any(y, 10)  # np -> eigen -> np
    y3 = incr_matrix_any(y2[0::2, 0::2], -33)  # np -> eigen -> np slice -> np -> eigen -> np
    y4 = even_rows(y3)  # numpy -> eigen slice -> (... y3)
    y5 = even_cols(y4)  # numpy -> eigen slice -> (... y4)
    y6 = incr_matrix_any(y5, 1000)  # numpy -> eigen -> (... y5)

    # Apply same mutations using just numpy:
    yexpect = np.array(range(100), dtype='float64').reshape(10, 10)
    yexpect += 10
    yexpect[0::2, 0::2] -= 33
    yexpect[0::4, 0::4] += 1000
    assert np.all(y6 == yexpect[0::4, 0::4])
    assert np.all(y5 == yexpect[0::4, 0::4])
    assert np.all(y4 == yexpect[0::4, 0::2])
    assert np.all(y3 == yexpect[0::2, 0::2])
    assert np.all(y2 == yexpect)
    assert np.all(y == yexpect)


def test_nocopy_wrapper():
    from pybind11_tests import get_elem, get_elem_nocopy, get_elem_rm_nocopy
    # get_elem requires a column-contiguous matrix reference, but should be
    # callable with other types of matrix (via copying):
    int_matrix_colmajor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], order='F')
    dbl_matrix_colmajor = np.array(int_matrix_colmajor, dtype='double', order='F', copy=True)
    int_matrix_rowmajor = np.array(int_matrix_colmajor, order='C', copy=True)
    dbl_matrix_rowmajor = np.array(int_matrix_rowmajor, dtype='double', order='C', copy=True)

    # All should be callable via get_elem:
    assert get_elem(int_matrix_colmajor) == 8
    assert get_elem(dbl_matrix_colmajor) == 8
    assert get_elem(int_matrix_rowmajor) == 8
    assert get_elem(dbl_matrix_rowmajor) == 8

    # All but the second should fail with get_elem_nocopy:
    with pytest.raises(TypeError) as excinfo:
        get_elem_nocopy(int_matrix_colmajor)
    assert ('get_elem_nocopy(): incompatible function arguments.' in str(excinfo.value) and
            ', flags.f_contiguous' in str(excinfo.value))
    assert get_elem_nocopy(dbl_matrix_colmajor) == 8
    with pytest.raises(TypeError) as excinfo:
        get_elem_nocopy(int_matrix_rowmajor)
    assert ('get_elem_nocopy(): incompatible function arguments.' in str(excinfo.value) and
            ', flags.f_contiguous' in str(excinfo.value))
    with pytest.raises(TypeError) as excinfo:
        get_elem_nocopy(dbl_matrix_rowmajor)
    assert ('get_elem_nocopy(): incompatible function arguments.' in str(excinfo.value) and
            ', flags.f_contiguous' in str(excinfo.value))

    # For the row-major test, we take a long matrix in row-major, so only the third is allowed:
    with pytest.raises(TypeError) as excinfo:
        get_elem_rm_nocopy(int_matrix_colmajor)
    assert ('get_elem_rm_nocopy(): incompatible function arguments.' in str(excinfo.value) and
            ', flags.c_contiguous' in str(excinfo.value))
    with pytest.raises(TypeError) as excinfo:
        get_elem_rm_nocopy(dbl_matrix_colmajor)
    assert ('get_elem_rm_nocopy(): incompatible function arguments.' in str(excinfo.value) and
            ', flags.c_contiguous' in str(excinfo.value))
    assert get_elem_rm_nocopy(int_matrix_rowmajor) == 8
    with pytest.raises(TypeError) as excinfo:
        get_elem_rm_nocopy(dbl_matrix_rowmajor)
    assert ('get_elem_rm_nocopy(): incompatible function arguments.' in str(excinfo.value) and
            ', flags.c_contiguous' in str(excinfo.value))


def test_special_matrix_objects():
    from pybind11_tests import incr_diag, symmetric_upper, symmetric_lower

    assert np.all(incr_diag(7) == np.diag([1., 2, 3, 4, 5, 6, 7]))

    asymm = np.array([[ 1.,  2,  3,  4],
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


def test_dense_signature(doc):
    from pybind11_tests import double_col, double_row, double_complex, double_mat_rm

    assert doc(double_col) == """
        double_col(arg0: numpy.ndarray[float32[m, 1]]) -> numpy.ndarray[float32[m, 1]]
    """
    assert doc(double_row) == """
        double_row(arg0: numpy.ndarray[float32[1, n]]) -> numpy.ndarray[float32[1, n]]
    """
    assert doc(double_complex) == """
        double_complex(arg0: numpy.ndarray[complex64[m, 1]]) -> numpy.ndarray[complex64[m, 1]]
    """
    assert doc(double_mat_rm) == """
        double_mat_rm(arg0: numpy.ndarray[float32[m, n]]) -> numpy.ndarray[float32[m, n]]
    """


def test_named_arguments():
    from pybind11_tests import matrix_multiply

    a = np.array([[1.0, 2], [3, 4], [5, 6]])
    b = np.ones((2, 1))

    assert np.all(matrix_multiply(a, b) == np.array([[3.], [7], [11]]))
    assert np.all(matrix_multiply(A=a, B=b) == np.array([[3.], [7], [11]]))
    assert np.all(matrix_multiply(B=b, A=a) == np.array([[3.], [7], [11]]))

    with pytest.raises(ValueError) as excinfo:
        matrix_multiply(b, a)
    assert str(excinfo.value) == 'Nonconformable matrices!'

    with pytest.raises(ValueError) as excinfo:
        matrix_multiply(A=b, B=a)
    assert str(excinfo.value) == 'Nonconformable matrices!'

    with pytest.raises(ValueError) as excinfo:
        matrix_multiply(B=a, A=b)
    assert str(excinfo.value) == 'Nonconformable matrices!'


@pytest.requires_eigen_and_scipy
def test_sparse():
    from pybind11_tests import sparse_r, sparse_c, sparse_copy_r, sparse_copy_c

    assert_sparse_equal_ref(sparse_r())
    assert_sparse_equal_ref(sparse_c())
    assert_sparse_equal_ref(sparse_copy_r(sparse_r()))
    assert_sparse_equal_ref(sparse_copy_c(sparse_c()))
    assert_sparse_equal_ref(sparse_copy_r(sparse_c()))
    assert_sparse_equal_ref(sparse_copy_c(sparse_r()))


@pytest.requires_eigen_and_scipy
def test_sparse_signature(doc):
    from pybind11_tests import sparse_copy_r, sparse_copy_c

    assert doc(sparse_copy_r) == """
        sparse_copy_r(arg0: scipy.sparse.csr_matrix[float32]) -> scipy.sparse.csr_matrix[float32]
    """  # noqa: E501 line too long
    assert doc(sparse_copy_c) == """
        sparse_copy_c(arg0: scipy.sparse.csc_matrix[float32]) -> scipy.sparse.csc_matrix[float32]
    """  # noqa: E501 line too long


def test_issue738():
    from pybind11_tests import iss738_f1, iss738_f2

    assert np.all(iss738_f1(np.array([[1., 2, 3]])) == np.array([[1., 102, 203]]))
    assert np.all(iss738_f1(np.array([[1.], [2], [3]])) == np.array([[1.], [12], [23]]))

    assert np.all(iss738_f2(np.array([[1., 2, 3]])) == np.array([[1., 102, 203]]))
    assert np.all(iss738_f2(np.array([[1.], [2], [3]])) == np.array([[1.], [12], [23]]))


def test_custom_operator_new():
    """Using Eigen types as member variables requires a class-specific
    operator new with proper alignment"""
    from pybind11_tests import CustomOperatorNew

    o = CustomOperatorNew()
    np.testing.assert_allclose(o.a, 0.0)
    np.testing.assert_allclose(o.b.diagonal(), 1.0)
