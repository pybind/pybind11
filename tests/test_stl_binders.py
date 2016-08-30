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


def test_map_string_double():
    from pybind11_tests import MapStringDouble, UnorderedMapStringDouble

    m = MapStringDouble()
    m['a'] = 1
    m['b'] = 2.5

    keys = []
    for k in m: keys.append(k)
    assert keys == ['a', 'b']

    key_values = []
    for k, v in m.items(): key_values.append( (k, v) )
    assert key_values == [('a', 1), ('b', 2.5) ]

    assert str(m) == "MapStringDouble{a: 1, b: 2.5}"


    um = UnorderedMapStringDouble()
    um['ua'] = 1.1
    um['ub'] = 2.6

    keys = []
    for k in um: keys.append(k)
    assert sorted(keys) == ['ua', 'ub']

    key_values = []
    for k, v in um.items(): key_values.append( (k, v) )
    assert sorted(key_values) == [('ua', 1.1), ('ub', 2.6) ]

    str(um)


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
