# -*- coding: utf-8 -*-
import io
import struct

import pytest

import env  # noqa: F401

from pybind11_tests import buffers as m
from pybind11_tests import ConstructorStats

np = pytest.importorskip("numpy")


def test_from_python():
    with pytest.raises(RuntimeError) as excinfo:
        m.Matrix(np.array([1, 2, 3]))  # trying to assign a 1D array
    assert str(excinfo.value) == "Incompatible buffer format!"

    m3 = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    m4 = m.Matrix(m3)

    for i in range(m4.rows()):
        for j in range(m4.cols()):
            assert m3[i, j] == m4[i, j]

    cstats = ConstructorStats.get(m.Matrix)
    assert cstats.alive() == 1
    del m3, m4
    assert cstats.alive() == 0
    assert cstats.values() == ["2x3 matrix"]
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0  # Don't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0


# https://foss.heptapod.net/pypy/pypy/-/issues/2444
def test_to_python():
    mat = m.Matrix(5, 4)
    assert memoryview(mat).shape == (5, 4)

    assert mat[2, 3] == 0
    mat[2, 3] = 4.0
    mat[3, 2] = 7.0
    assert mat[2, 3] == 4
    assert mat[3, 2] == 7
    assert struct.unpack_from('f', mat, (3 * 4 + 2) * 4) == (7, )
    assert struct.unpack_from('f', mat, (2 * 4 + 3) * 4) == (4, )

    mat2 = np.array(mat, copy=False)
    assert mat2.shape == (5, 4)
    assert abs(mat2).sum() == 11
    assert mat2[2, 3] == 4 and mat2[3, 2] == 7
    mat2[2, 3] = 5
    assert mat2[2, 3] == 5

    cstats = ConstructorStats.get(m.Matrix)
    assert cstats.alive() == 1
    del mat
    pytest.gc_collect()
    assert cstats.alive() == 1
    del mat2  # holds a mat reference
    pytest.gc_collect()
    assert cstats.alive() == 0
    assert cstats.values() == ["5x4 matrix"]
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0  # Don't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0


def test_inherited_protocol():
    """SquareMatrix is derived from Matrix and inherits the buffer protocol"""

    matrix = m.SquareMatrix(5)
    assert memoryview(matrix).shape == (5, 5)
    assert np.asarray(matrix).shape == (5, 5)


def test_pointer_to_member_fn():
    for cls in [m.Buffer, m.ConstBuffer, m.DerivedBuffer]:
        buf = cls()
        buf.value = 0x12345678
        value = struct.unpack('i', bytearray(buf))[0]
        assert value == 0x12345678


def test_readonly_buffer():
    buf = m.BufferReadOnly(0x64)
    view = memoryview(buf)
    assert view[0] == b'd' if env.PY2 else 0x64
    assert view.readonly


def test_selective_readonly_buffer():
    buf = m.BufferReadOnlySelect()

    memoryview(buf)[0] = b'd' if env.PY2 else 0x64
    assert buf.value == 0x64

    io.BytesIO(b'A').readinto(buf)
    assert buf.value == ord(b'A')

    buf.readonly = True
    with pytest.raises(TypeError):
        memoryview(buf)[0] = b'\0' if env.PY2 else 0
    with pytest.raises(TypeError):
        io.BytesIO(b'1').readinto(buf)
