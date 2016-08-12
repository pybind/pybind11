

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
