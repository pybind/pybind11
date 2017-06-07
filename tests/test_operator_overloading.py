import pytest
from pybind11_tests import ConstructorStats


def test_operator_overloading():
    from pybind11_tests.operators import Vector2, Vector

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


def test_operators_notimplemented():
    """#393: need to return NotSupported to ensure correct arithmetic operator behavior"""
    from pybind11_tests.operators import C1, C2

    c1, c2 = C1(), C2()
    assert c1 + c1 == 11
    assert c2 + c2 == 22
    assert c2 + c1 == 21
    assert c1 + c2 == 12


def test_nested():
    """#328: first member in a class can't be used in operators"""
    from pybind11_tests.operators import NestA, NestB, NestC, get_NestA, get_NestB, get_NestC

    a = NestA()
    b = NestB()
    c = NestC()

    a += 10
    assert get_NestA(a) == 13
    b.a += 100
    assert get_NestA(b.a) == 103
    c.b.a += 1000
    assert get_NestA(c.b.a) == 1003
    b -= 1
    assert get_NestB(b) == 3
    c.b -= 3
    assert get_NestB(c.b) == 1
    c *= 7
    assert get_NestC(c) == 35

    abase = a.as_base()
    assert abase.value == -2
    a.as_base().value += 44
    assert abase.value == 42
    assert c.b.a.as_base().value == -2
    c.b.a.as_base().value += 44
    assert c.b.a.as_base().value == 42

    del c
    pytest.gc_collect()
    del a  # Should't delete while abase is still alive
    pytest.gc_collect()

    assert abase.value == 42
    del abase, b
    pytest.gc_collect()
