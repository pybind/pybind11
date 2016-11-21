import pytest
import sys

with pytest.suppress(ImportError):
    import numpy as np


def test_vector_int():
    from pybind11_tests import VectorInt

    v_int = VectorInt([0, 0])
    assert len(v_int) == 2
    assert bool(v_int) is True

    v_int2 = VectorInt([0, 0])
    assert v_int == v_int2
    v_int2[1] = 1
    assert v_int != v_int2

    v_int2.append(2)
    v_int2.append(3)
    v_int2.insert(0, 1)
    v_int2.insert(0, 2)
    v_int2.insert(0, 3)
    assert str(v_int2) == "VectorInt[3, 2, 1, 0, 1, 2, 3]"

    v_int.append(99)
    v_int2[2:-2] = v_int
    assert v_int2 == VectorInt([3, 2, 0, 0, 99, 2, 3])
    del v_int2[1:3]
    assert v_int2 == VectorInt([3, 0, 99, 2, 3])
    del v_int2[0]
    assert v_int2 == VectorInt([0, 99, 2, 3])


@pytest.unsupported_on_pypy
def test_vector_buffer():
    from pybind11_tests import VectorUChar, create_undeclstruct
    b = bytearray([1, 2, 3, 4])
    v = VectorUChar(b)
    assert v[1] == 2
    v[2] = 5
    m = memoryview(v)  # We expose the buffer interface
    if sys.version_info.major > 2:
        assert m[2] == 5
        m[2] = 6
    else:
        assert m[2] == '\x05'
        m[2] = '\x06'
    assert v[2] == 6

    with pytest.raises(RuntimeError):
        create_undeclstruct()  # Undeclared struct contents, no buffer interface


@pytest.requires_numpy
def test_vector_buffer_numpy():
    from pybind11_tests import VectorInt, VectorStruct, get_vectorstruct

    a = np.array([1, 2, 3, 4], dtype=np.int32)
    with pytest.raises(TypeError):
        VectorInt(a)

    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uintc)
    v = VectorInt(a[0, :])
    assert len(v) == 4
    assert v[2] == 3
    m = np.asarray(v)
    m[2] = 5
    assert v[2] == 5

    v = VectorInt(a[:, 1])
    assert len(v) == 3
    assert v[2] == 10

    v = get_vectorstruct()
    assert v[0].x == 5
    m = np.asarray(v)
    m[1]['x'] = 99
    assert v[1].x == 99

    v = VectorStruct(np.zeros(3, dtype=np.dtype([('w', 'bool'), ('x', 'I'),
                                                 ('y', 'float64'), ('z', 'bool')], align=True)))
    assert len(v) == 3


def test_vector_custom():
    from pybind11_tests import El, VectorEl, VectorVectorEl

    v_a = VectorEl()
    v_a.append(El(1))
    v_a.append(El(2))
    assert str(v_a) == "VectorEl[El{1}, El{2}]"

    vv_a = VectorVectorEl()
    vv_a.append(v_a)
    vv_b = vv_a[0]
    assert str(vv_b) == "VectorEl[El{1}, El{2}]"


def test_vector_bool():
    from pybind11_tests import VectorBool

    vv_c = VectorBool()
    for i in range(10):
        vv_c.append(i % 2 == 0)
    for i in range(10):
        assert vv_c[i] == (i % 2 == 0)
    assert str(vv_c) == "VectorBool[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]"


def test_map_string_double():
    from pybind11_tests import MapStringDouble, UnorderedMapStringDouble

    m = MapStringDouble()
    m['a'] = 1
    m['b'] = 2.5

    assert list(m) == ['a', 'b']
    assert list(m.items()) == [('a', 1), ('b', 2.5)]
    assert str(m) == "MapStringDouble{a: 1, b: 2.5}"

    um = UnorderedMapStringDouble()
    um['ua'] = 1.1
    um['ub'] = 2.6

    assert sorted(list(um)) == ['ua', 'ub']
    assert sorted(list(um.items())) == [('ua', 1.1), ('ub', 2.6)]
    assert "UnorderedMapStringDouble" in str(um)


def test_map_string_double_const():
    from pybind11_tests import MapStringDoubleConst, UnorderedMapStringDoubleConst

    mc = MapStringDoubleConst()
    mc['a'] = 10
    mc['b'] = 20.5
    assert str(mc) == "MapStringDoubleConst{a: 10, b: 20.5}"

    umc = UnorderedMapStringDoubleConst()
    umc['a'] = 11
    umc['b'] = 21.5

    str(umc)


def test_noncopyable_vector():
    from pybind11_tests import get_vnc

    vnc = get_vnc(5)
    for i in range(0, 5):
        assert vnc[i].value == i + 1

    for i, j in enumerate(vnc, start=1):
        assert j.value == i


def test_noncopyable_deque():
    from pybind11_tests import get_dnc

    dnc = get_dnc(5)
    for i in range(0, 5):
        assert dnc[i].value == i + 1

    i = 1
    for j in dnc:
        assert(j.value == i)
        i += 1


def test_noncopyable_map():
    from pybind11_tests import get_mnc

    mnc = get_mnc(5)
    for i in range(1, 6):
        assert mnc[i].value == 10 * i

    vsum = 0
    for k, v in mnc.items():
        assert v.value == 10 * k
        vsum += v.value

    assert vsum == 150


def test_noncopyable_unordered_map():
    from pybind11_tests import get_umnc

    mnc = get_umnc(5)
    for i in range(1, 6):
        assert mnc[i].value == 10 * i

    vsum = 0
    for k, v in mnc.items():
        assert v.value == 10 * k
        vsum += v.value

    assert vsum == 150
