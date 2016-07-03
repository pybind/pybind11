import pytest

def test_implicit_conversion():

    from pybind11_tests import (ExIC_A, ExIC_B, ExIC_C, ExIC_D, ExIC_F,
                                as_double, as_string, double_exICe, double_exICf,
                                ConstructorStats, cstats_ExIC_E)

    # ExIC_A is declared cpp convertible to double; ExIC_B is a registered subclass of ExIC_A,
    # and ExIC_C is a registered subclass of ExIC_B.  All should be convertible to double
    # through ExIC_A's base class convertibility.
    assert as_double(ExIC_A()) == 42
    phi = (5 ** (1/2.0) + 1) / 2 # 1.6180339...
    pi = 3.14159265358979323846
    e = 2.71828182845904523536
    assert as_double(ExIC_A(phi)) == phi
    assert as_double(ExIC_B()) == 42
    assert as_double(ExIC_C()) == pi
    assert as_string(ExIC_C()) == "pi"
    assert as_double(ExIC_D()) == e
    assert as_string(ExIC_D()) == "e"

    with pytest.raises(TypeError) as excinfo:
        double_exICe(ExIC_A())
        raise RuntimeError("ExIC_A should not be implicitly convertible to ExIC_E")

    assert double_exICe(ExIC_B()) == 84
    assert double_exICe(ExIC_C()) == 2*pi
    assert double_exICe(ExIC_D()) == 3*e

    assert double_exICf(ExIC_F()) == 99
    assert double_exICf(ExIC_A(0.25)) == 250
    assert -1e-10 < double_exICf(ExIC_C()) - 1000*pi < 1e-10

    with pytest.raises(TypeError) as excinfo:
        double_exICf(ExIC_F(ExIC_A(4)))
        raise RuntimeError("BAD: ExIC_F conversion constructor (from ExIC_A) should not have been exposed to python")

    cstats = ConstructorStats.get(ExIC_A)
    assert cstats.alive() == 0
    expected_values = ['double conversion operator', '1.61803', 'double conversion operator', 'double conversion operator', 'double conversion operator', '0.25', 'double conversion operator', '4']
    assert cstats.values() == expected_values
    assert cstats.default_constructions == 11
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0

    cstats = ConstructorStats.get(ExIC_B)
    assert cstats.alive() == 0
    expected_values = ['ExIC_E conversion operator', 'ExIC_E conversion operator']
    assert cstats.values() == expected_values
    assert cstats.default_constructions == 6
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0

    cstats = ConstructorStats.get(ExIC_C)
    assert cstats.alive() == 0
    expected_values = []
    assert cstats.values() == expected_values
    assert cstats.default_constructions == 4
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0

    cstats = ConstructorStats.get(ExIC_D)
    assert cstats.alive() == 0
    expected_values = []
    assert cstats.values() == expected_values
    assert cstats.default_constructions == 3
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0

    cstats = cstats_ExIC_E()
    assert cstats.alive() == 0
    expected_values = ['double constructor', '84', 'double conversion operator', 'double constructor', '6.28319', 'double conversion operator', 'ExIC_D conversion constructor', 'double conversion operator']
    assert cstats.values() == expected_values
    assert cstats.default_constructions == 0
    assert cstats.copy_constructions == 0
    assert cstats.move_constructions >= 3
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0


def test_implicit_conversion_order():
    from pybind11_tests import ExIC_G1, ExIC_G2, ExIC_G3, ExIC_G4, as_long

    assert as_long(ExIC_G1()) == 111
    assert as_long(ExIC_G2()) == 222
    assert as_long(ExIC_G3()) == 333
    assert as_long(ExIC_G4()) == 444
