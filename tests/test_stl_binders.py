# -*- coding: utf-8 -*-
import pytest

import env
from pybind11_tests import stl_binders as m


def test_vector_int():
    v_int = m.VectorInt([0, 0])
    assert len(v_int) == 2
    assert bool(v_int) is True

    # test construction from a generator
    v_int1 = m.VectorInt(x for x in range(5))
    assert v_int1 == m.VectorInt([0, 1, 2, 3, 4])

    v_int2 = m.VectorInt([0, 0])
    assert v_int == v_int2
    v_int2[1] = 1
    assert v_int != v_int2

    v_int2.append(2)
    v_int2.insert(0, 1)
    v_int2.insert(0, 2)
    v_int2.insert(0, 3)
    v_int2.insert(6, 3)
    assert str(v_int2) == "VectorInt[3, 2, 1, 0, 1, 2, 3]"
    with pytest.raises(IndexError):
        v_int2.insert(8, 4)

    v_int.append(99)
    v_int2[2:-2] = v_int
    assert v_int2 == m.VectorInt([3, 2, 0, 0, 99, 2, 3])
    del v_int2[1:3]
    assert v_int2 == m.VectorInt([3, 0, 99, 2, 3])
    del v_int2[0]
    assert v_int2 == m.VectorInt([0, 99, 2, 3])

    v_int2.extend(m.VectorInt([4, 5]))
    assert v_int2 == m.VectorInt([0, 99, 2, 3, 4, 5])

    v_int2.extend([6, 7])
    assert v_int2 == m.VectorInt([0, 99, 2, 3, 4, 5, 6, 7])

    # test error handling, and that the vector is unchanged
    with pytest.raises(RuntimeError):
        v_int2.extend([8, "a"])

    assert v_int2 == m.VectorInt([0, 99, 2, 3, 4, 5, 6, 7])

    # test extending from a generator
    v_int2.extend(x for x in range(5))
    assert v_int2 == m.VectorInt([0, 99, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4])

    # test negative indexing
    assert v_int2[-1] == 4

    # insert with negative index
    v_int2.insert(-1, 88)
    assert v_int2 == m.VectorInt([0, 99, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 88, 4])

    # delete negative index
    del v_int2[-1]
    assert v_int2 == m.VectorInt([0, 99, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 88])

    v_int2.clear()
    assert len(v_int2) == 0


# Older PyPy's failed here, related to the PyPy's buffer protocol.
def test_vector_buffer():
    b = bytearray([1, 2, 3, 4])
    v = m.VectorUChar(b)
    assert v[1] == 2
    v[2] = 5
    mv = memoryview(v)  # We expose the buffer interface
    if not env.PY2:
        assert mv[2] == 5
        mv[2] = 6
    else:
        assert mv[2] == "\x05"
        mv[2] = "\x06"
    assert v[2] == 6

    if not env.PY2:
        mv = memoryview(b)
        v = m.VectorUChar(mv[::2])
        assert v[1] == 3

    with pytest.raises(RuntimeError) as excinfo:
        m.create_undeclstruct()  # Undeclared struct contents, no buffer interface
    assert "NumPy type info missing for " in str(excinfo.value)


def test_vector_buffer_numpy():
    np = pytest.importorskip("numpy")
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    with pytest.raises(TypeError):
        m.VectorInt(a)

    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uintc)
    v = m.VectorInt(a[0, :])
    assert len(v) == 4
    assert v[2] == 3
    ma = np.asarray(v)
    ma[2] = 5
    assert v[2] == 5

    v = m.VectorInt(a[:, 1])
    assert len(v) == 3
    assert v[2] == 10

    v = m.get_vectorstruct()
    assert v[0].x == 5
    ma = np.asarray(v)
    ma[1]["x"] = 99
    assert v[1].x == 99

    v = m.VectorStruct(
        np.zeros(
            3,
            dtype=np.dtype(
                [("w", "bool"), ("x", "I"), ("y", "float64"), ("z", "bool")], align=True
            ),
        )
    )
    assert len(v) == 3

    b = np.array([1, 2, 3, 4], dtype=np.uint8)
    v = m.VectorUChar(b[::2])
    assert v[1] == 3


def test_vector_bool():
    import pybind11_cross_module_tests as cm

    vv_c = cm.VectorBool()
    for i in range(10):
        vv_c.append(i % 2 == 0)
    for i in range(10):
        assert vv_c[i] == (i % 2 == 0)
    assert str(vv_c) == "VectorBool[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]"


def test_vector_custom():
    v_a = m.VectorEl()
    v_a.append(m.El(1))
    v_a.append(m.El(2))
    assert str(v_a) == "VectorEl[El{1}, El{2}]"

    vv_a = m.VectorVectorEl()
    vv_a.append(v_a)
    vv_b = vv_a[0]
    assert str(vv_b) == "VectorEl[El{1}, El{2}]"


def test_map_string_double():
    mm = m.MapStringDouble()
    mm["a"] = 1
    mm["b"] = 2.5

    assert list(mm) == ["a", "b"]
    assert str(mm) == "MapStringDouble{a: 1, b: 2.5}"
    assert "b" in mm
    assert "c" not in mm
    assert 123 not in mm

    # Check that keys, values, items are views, not merely iterable
    keys = mm.keys()
    values = mm.values()
    items = mm.items()
    assert list(keys) == ["a", "b"]
    assert len(keys) == 2
    assert "a" in keys
    assert "c" not in keys
    assert 123 not in keys
    assert list(items) == [("a", 1), ("b", 2.5)]
    assert len(items) == 2
    assert ("b", 2.5) in items
    assert "hello" not in items
    assert ("b", 2.5, None) not in items
    assert list(values) == [1, 2.5]
    assert len(values) == 2
    assert 1 in values
    assert 2 not in values
    # Check that views update when the map is updated
    mm["c"] = -1
    assert list(keys) == ["a", "b", "c"]
    assert list(values) == [1, 2.5, -1]
    assert list(items) == [("a", 1), ("b", 2.5), ("c", -1)]

    um = m.UnorderedMapStringDouble()
    um["ua"] = 1.1
    um["ub"] = 2.6

    assert sorted(list(um)) == ["ua", "ub"]
    assert list(um.keys()) == list(um)
    assert sorted(list(um.items())) == [("ua", 1.1), ("ub", 2.6)]
    assert list(zip(um.keys(), um.values())) == list(um.items())
    assert "UnorderedMapStringDouble" in str(um)


def test_map_string_double_const():
    mc = m.MapStringDoubleConst()
    mc["a"] = 10
    mc["b"] = 20.5
    assert str(mc) == "MapStringDoubleConst{a: 10, b: 20.5}"

    umc = m.UnorderedMapStringDoubleConst()
    umc["a"] = 11
    umc["b"] = 21.5

    str(umc)


def test_noncopyable_containers():
    # std::vector
    vnc = m.get_vnc(5)
    for i in range(0, 5):
        assert vnc[i].value == i + 1

    for i, j in enumerate(vnc, start=1):
        assert j.value == i

    # std::deque
    dnc = m.get_dnc(5)
    for i in range(0, 5):
        assert dnc[i].value == i + 1

    i = 1
    for j in dnc:
        assert j.value == i
        i += 1

    # std::map
    mnc = m.get_mnc(5)
    for i in range(1, 6):
        assert mnc[i].value == 10 * i

    vsum = 0
    for k, v in mnc.items():
        assert v.value == 10 * k
        vsum += v.value

    assert vsum == 150

    # std::unordered_map
    mnc = m.get_umnc(5)
    for i in range(1, 6):
        assert mnc[i].value == 10 * i

    vsum = 0
    for k, v in mnc.items():
        assert v.value == 10 * k
        vsum += v.value

    assert vsum == 150

    # nested std::map<std::vector>
    nvnc = m.get_nvnc(5)
    for i in range(1, 6):
        for j in range(0, 5):
            assert nvnc[i][j].value == j + 1

    # Note: maps do not have .values()
    for _, v in nvnc.items():
        for i, j in enumerate(v, start=1):
            assert j.value == i

    # nested std::map<std::map>
    nmnc = m.get_nmnc(5)
    for i in range(1, 6):
        for j in range(10, 60, 10):
            assert nmnc[i][j].value == 10 * j

    vsum = 0
    for _, v_o in nmnc.items():
        for k_i, v_i in v_o.items():
            assert v_i.value == 10 * k_i
            vsum += v_i.value

    assert vsum == 7500

    # nested std::unordered_map<std::unordered_map>
    numnc = m.get_numnc(5)
    for i in range(1, 6):
        for j in range(10, 60, 10):
            assert numnc[i][j].value == 10 * j

    vsum = 0
    for _, v_o in numnc.items():
        for k_i, v_i in v_o.items():
            assert v_i.value == 10 * k_i
            vsum += v_i.value

    assert vsum == 7500


def test_map_delitem():
    mm = m.MapStringDouble()
    mm["a"] = 1
    mm["b"] = 2.5

    assert list(mm) == ["a", "b"]
    assert list(mm.items()) == [("a", 1), ("b", 2.5)]
    del mm["a"]
    assert list(mm) == ["b"]
    assert list(mm.items()) == [("b", 2.5)]

    um = m.UnorderedMapStringDouble()
    um["ua"] = 1.1
    um["ub"] = 2.6

    assert sorted(list(um)) == ["ua", "ub"]
    assert sorted(list(um.items())) == [("ua", 1.1), ("ub", 2.6)]
    del um["ua"]
    assert sorted(list(um)) == ["ub"]
    assert sorted(list(um.items())) == [("ub", 2.6)]
