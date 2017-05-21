def test_operator_overloading():
    from pybind11_tests import Vector2, Vector, ConstructorStats

    v1 = Vector2(1, 2)
    v2 = Vector(3, -1)
    assert str(v1) == "[1.000000, 2.000000]"
    assert str(v2) == "[3.000000, -1.000000]"

    assert str(v1 + v2) == "[4.000000, 1.000000]"
    assert str(v1 - v2) == "[-2.000000, 3.000000]"
    assert str(v1 - 8) == "[-7.000000, -6.000000]"
    assert str(v1 + 8) == "[9.000000, 10.000000]"
    assert str(v1 * 8) == "[8.000000, 16.000000]"
    assert str(v1 / 8) == "[0.125000, 0.250000]"
    assert str(8 - v1) == "[7.000000, 6.000000]"
    assert str(8 + v1) == "[9.000000, 10.000000]"
    assert str(8 * v1) == "[8.000000, 16.000000]"
    assert str(8 / v1) == "[8.000000, 4.000000]"
    assert str(v1 * v2) == "[3.000000, -2.000000]"
    assert str(v2 / v1) == "[3.000000, -0.500000]"

    v1 += 2 * v2
    assert str(v1) == "[7.000000, 0.000000]"
    v1 -= v2
    assert str(v1) == "[4.000000, 1.000000]"
    v1 *= 2
    assert str(v1) == "[8.000000, 2.000000]"
    v1 /= 16
    assert str(v1) == "[0.500000, 0.125000]"
    v1 *= v2
    assert str(v1) == "[1.500000, -0.125000]"
    v2 /= v1
    assert str(v2) == "[2.000000, 8.000000]"

    cstats = ConstructorStats.get(Vector2)
    assert cstats.alive() == 2
    del v1
    assert cstats.alive() == 1
    del v2
    assert cstats.alive() == 0
    assert cstats.values() == ['[1.000000, 2.000000]', '[3.000000, -1.000000]',
                               '[4.000000, 1.000000]', '[-2.000000, 3.000000]',
                               '[-7.000000, -6.000000]', '[9.000000, 10.000000]',
                               '[8.000000, 16.000000]', '[0.125000, 0.250000]',
                               '[7.000000, 6.000000]', '[9.000000, 10.000000]',
                               '[8.000000, 16.000000]', '[8.000000, 4.000000]',
                               '[3.000000, -2.000000]', '[3.000000, -0.500000]',
                               '[6.000000, -2.000000]']
    assert cstats.default_constructions == 0
    assert cstats.copy_constructions == 0
    assert cstats.move_constructions >= 10
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0
