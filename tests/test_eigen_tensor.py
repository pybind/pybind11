import pytest

np = pytest.importorskip("numpy")
m = pytest.importorskip("pybind11_tests.eigen_tensor")

tensor_ref = np.array(
    [
        [[0.0, 15.0], [3.0, 18.0], [6.0, 21.0], [9.0, 24.0], [12.0, 27.0]],
        [[1.0, 16.0], [4.0, 19.0], [7.0, 22.0], [10.0, 25.0], [13.0, 28.0]],
        [[2.0, 17.0], [5.0, 20.0], [8.0, 23.0], [11.0, 26.0], [14.0, 29.0]],
    ],
)

indices = (2, 3, 1)


def assert_equal_tensor_ref(mat, writeable=True, modified=0):
    assert mat.flags.writeable == writeable

    copy = np.array(tensor_ref)
    if modified != 0:
        copy[indices] = modified

    np.testing.assert_array_equal(mat, copy)


def test_convert_tensor_to_py():
    assert_equal_tensor_ref(m.copy_tensor())
    assert_equal_tensor_ref(m.copy_fixed_tensor())
    assert_equal_tensor_ref(m.copy_const_tensor())

    assert_equal_tensor_ref(m.move_tensor())
    assert_equal_tensor_ref(m.move_fixed_tensor())

    assert_equal_tensor_ref(m.take_tensor())
    assert_equal_tensor_ref(m.take_fixed_tensor())

    assert_equal_tensor_ref(m.reference_tensor())
    assert_equal_tensor_ref(m.reference_tensor_v2())
    assert_equal_tensor_ref(m.reference_fixed_tensor())

    assert_equal_tensor_ref(m.reference_view_of_tensor())
    assert_equal_tensor_ref(m.reference_view_of_fixed_tensor())
    assert_equal_tensor_ref(m.reference_const_tensor(), writeable=False)
    assert_equal_tensor_ref(m.reference_const_tensor_v2(), writeable=False)
    

def test_bad_cpp_to_python_casts():
    with pytest.raises(Exception):
        m.reference_tensor_internal()
    
    with pytest.raises(Exception):
        m.move_const_tensor()

    with pytest.raises(Exception):
        m.take_const_tensor()


def test_bad_python_to_cpp_casts():
    with pytest.raises(TypeError):
        m.round_trip_tensor(np.zeros((2, 3)))

    with pytest.raises(TypeError):
        m.round_trip_tensor(np.zeros(dtype=np.str_, shape=(2, 3, 1)))

    # Shape, dtype and the order need to be correct for a TensorMap cast
    with pytest.raises(TypeError):
        m.round_trip_view_tensor(np.zeros((3, 5, 2), dtype=np.float64, order="C"))
    with pytest.raises(TypeError):
        m.round_trip_view_tensor(np.zeros((3, 5, 2), dtype=np.float32, order="F"))
    with pytest.raises(TypeError):
        m.round_trip_view_tensor(np.zeros((3, 5), dtype=np.float64, order="F"))
    with pytest.raises(TypeError):
        temp = np.zeros((3, 5, 2), dtype=np.float64, order="F")
        temp.setflags(write=False)
        m.round_trip_view_tensor(temp)

def test_references_actually_refer():
    a = m.reference_tensor()
    temp = a[indices]
    a[indices] = 100
    assert_equal_tensor_ref(m.copy_const_tensor(), modified=100)
    a[indices] = temp
    assert_equal_tensor_ref(m.copy_const_tensor())

    a = m.reference_view_of_tensor()
    a[indices] = 100
    assert_equal_tensor_ref(m.copy_const_tensor(), modified=100)
    a[indices] = temp
    assert_equal_tensor_ref(m.copy_const_tensor())


def test_round_trip():
    assert_equal_tensor_ref(m.round_trip_tensor(tensor_ref))
    assert_equal_tensor_ref(m.round_trip_aligned_view_tensor(m.reference_tensor()))
    
    copy = np.array(tensor_ref, dtype=np.float64, order="F")
    assert_equal_tensor_ref(m.round_trip_view_tensor(copy))
    copy.setflags(write=False)
    assert_equal_tensor_ref(m.round_trip_const_view_tensor(copy))

def test_round_trip_references_actually_refer():
    # Need to create a copy that matches the type on the C side
    copy = np.array(tensor_ref, dtype=np.float64, order="F")
    a = m.round_trip_view_tensor(copy)
    temp = a[indices]
    a[indices] = 100
    assert_equal_tensor_ref(copy, modified=100)
    a[indices] = temp
    assert_equal_tensor_ref(copy)


def test_doc_string(doc):
    assert (
        doc(m.copy_tensor)
        == "copy_tensor() -> numpy.ndarray[numpy.float64[?, ?, ?], flags.f_contiguous]"
    )
    assert (
        doc(m.copy_fixed_tensor)
        == "copy_fixed_tensor() -> numpy.ndarray[numpy.float64[3, 5, 2], flags.f_contiguous]"
    )
    assert (
        doc(m.reference_const_tensor)
        == "reference_const_tensor() -> numpy.ndarray[numpy.float64[?, ?, ?], flags.f_contiguous]"
    )
    assert (
        doc(m.round_trip_view_tensor)
        == "round_trip_view_tensor(arg0: numpy.ndarray[numpy.float64[?, ?, ?], flags.writeable, flags.f_contiguous]) -> numpy.ndarray[numpy.float64[?, ?, ?], flags.writeable, flags.f_contiguous]"
    )
    assert (
        doc(m.round_trip_const_view_tensor)
        == "round_trip_const_view_tensor(arg0: numpy.ndarray[numpy.float64[?, ?, ?], flags.f_contiguous]) -> numpy.ndarray[numpy.float64[?, ?, ?], flags.f_contiguous]"
    )