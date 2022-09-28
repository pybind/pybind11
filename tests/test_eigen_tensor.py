import pytest

from pybind11_tests import ConstructorStats

np = pytest.importorskip("numpy")
m = pytest.importorskip("pybind11_tests.eigen_tensor")

tensor_ref = np.array(
    [
        [[0, 3]],
        [[1, 4]],
        [[2, 5]],
    ]
)


def assert_equal_tensor_ref(mat, writeable=True, modified=0):
    assert mat.flags.writeable == writeable

    if modified != 0:
        tensor_ref[0, 0, 0] = modified

    np.testing.assert_array_equal(mat, tensor_ref)

    tensor_ref[0, 0, 0] = 0


def test_convert_tensor_to_py():
    assert_equal_tensor_ref(m.copy_global_tensor())
    assert_equal_tensor_ref(m.copy_fixed_global_tensor())
    assert_equal_tensor_ref(m.copy_const_global_tensor())

    assert_equal_tensor_ref(m.reference_global_tensor())
    assert_equal_tensor_ref(m.reference_view_of_global_tensor())
    assert_equal_tensor_ref(m.reference_view_of_fixed_global_tensor())
    assert_equal_tensor_ref(m.reference_const_global_tensor(), writeable=False)


def test_references_actually_refer():
    a = m.reference_global_tensor()
    a[0, 0, 0] = 100
    assert_equal_tensor_ref(m.copy_const_global_tensor(), modified=100)
    a[0, 0, 0] = 0
    assert_equal_tensor_ref(m.copy_const_global_tensor())

    a = m.reference_view_of_global_tensor()
    a[0, 0, 0] = 100
    assert_equal_tensor_ref(m.copy_const_global_tensor(), modified=100)
    a[0, 0, 0] = 0
    assert_equal_tensor_ref(m.copy_const_global_tensor())


def test_round_trip():
    assert_equal_tensor_ref(m.round_trip_tensor(tensor_ref))


def test_round_trip_references_actually_refer():
    # Need to create a copy that matches the type on the C side
    copy = np.array(tensor_ref, dtype=np.float64, order="F")
    a = m.round_trip_view_tensor(copy)
    a[0, 0, 0] = 100
    assert_equal_tensor_ref(copy, modified=100)
    a[0, 0, 0] = 0
    assert_equal_tensor_ref(copy)
