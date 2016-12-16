import pytest
from pybind11_tests import Matrix, ConstructorStats

with pytest.suppress(ImportError):
    import numpy as np


@pytest.requires_numpy
def test_from_python():
    with pytest.raises(RuntimeError) as excinfo:
        Matrix(np.array([1, 2, 3]))  # trying to assign a 1D array
    assert str(excinfo.value) == "Incompatible buffer format!"

    m3 = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    m4 = Matrix(m3)

    for i in range(m4.rows()):
        for j in range(m4.cols()):
            assert m3[i, j] == m4[i, j]

    cstats = ConstructorStats.get(Matrix)
    assert cstats.alive() == 1
    del m3, m4
    assert cstats.alive() == 0
    assert cstats.values() == ["2x3 matrix"]
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0  # Don't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0


# PyPy: Memory leak in the "np.array(m, copy=False)" call
# https://bitbucket.org/pypy/pypy/issues/2444
@pytest.unsupported_on_pypy
@pytest.requires_numpy
def test_to_python():
    m = Matrix(5, 5)

    assert m[2, 3] == 0
    m[2, 3] = 4
    assert m[2, 3] == 4

    m2 = np.array(m, copy=False)
    assert m2.shape == (5, 5)
    assert abs(m2).sum() == 4
    assert m2[2, 3] == 4
    m2[2, 3] = 5
    assert m2[2, 3] == 5

    cstats = ConstructorStats.get(Matrix)
    assert cstats.alive() == 1
    del m
    pytest.gc_collect()
    assert cstats.alive() == 1
    del m2  # holds an m reference
    pytest.gc_collect()
    assert cstats.alive() == 0
    assert cstats.values() == ["5x5 matrix"]
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0  # Don't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0
