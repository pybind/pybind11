import pytest
from pybind11_tests import ConstructorStats


def test_multiple_inheritance_cpp():
    from pybind11_tests import MIType

    mt = MIType(3, 4)

    assert mt.foo() == 3
    assert mt.bar() == 4


def test_multiple_inheritance_mix1():
    from pybind11_tests import Base2

    class Base1:
        def __init__(self, i):
            self.i = i

        def foo(self):
            return self.i

    class MITypePy(Base1, Base2):
        def __init__(self, i, j):
            Base1.__init__(self, i)
            Base2.__init__(self, j)

    mt = MITypePy(3, 4)

    assert mt.foo() == 3
    assert mt.bar() == 4


def test_multiple_inheritance_mix2():
    from pybind11_tests import Base1

    class Base2:
        def __init__(self, i):
            self.i = i

        def bar(self):
            return self.i

    class MITypePy(Base1, Base2):
        def __init__(self, i, j):
            Base1.__init__(self, i)
            Base2.__init__(self, j)

    mt = MITypePy(3, 4)

    assert mt.foo() == 3
    assert mt.bar() == 4


def test_multiple_inheritance_error():
    """Inheriting from multiple C++ bases in Python is not supported"""
    from pybind11_tests import Base1, Base2

    with pytest.raises(TypeError) as excinfo:
        # noinspection PyUnusedLocal
        class MI(Base1, Base2):
            pass
    assert "Can't inherit from multiple C++ classes in Python" in str(excinfo.value)


def test_multiple_inheritance_virtbase():
    from pybind11_tests import Base12a, bar_base2a, bar_base2a_sharedptr

    class MITypePy(Base12a):
        def __init__(self, i, j):
            Base12a.__init__(self, i, j)

    mt = MITypePy(3, 4)
    assert mt.bar() == 4
    assert bar_base2a(mt) == 4
    assert bar_base2a_sharedptr(mt) == 4


def test_mi_static_properties():
    """Mixing bases with and without static properties should be possible
     and the result should be independent of base definition order"""
    from pybind11_tests import mi

    for d in (mi.VanillaStaticMix1(), mi.VanillaStaticMix2()):
        assert d.vanilla() == "Vanilla"
        assert d.static_func1() == "WithStatic1"
        assert d.static_func2() == "WithStatic2"
        assert d.static_func() == d.__class__.__name__

        mi.WithStatic1.static_value1 = 1
        mi.WithStatic2.static_value2 = 2
        assert d.static_value1 == 1
        assert d.static_value2 == 2
        assert d.static_value == 12

        d.static_value1 = 0
        assert d.static_value1 == 0
        d.static_value2 = 0
        assert d.static_value2 == 0
        d.static_value = 0
        assert d.static_value == 0


@pytest.unsupported_on_pypy
def test_mi_dynamic_attributes():
    """Mixing bases with and without dynamic attribute support"""
    from pybind11_tests import mi

    for d in (mi.VanillaDictMix1(), mi.VanillaDictMix2()):
        d.dynamic = 1
        assert d.dynamic == 1


def test_mi_unaligned_base():
    """Returning an offset (non-first MI) base class pointer should recognize the instance"""
    from pybind11_tests import I801C, I801D, i801b1_c, i801b2_c, i801b1_d, i801b2_d

    n_inst = ConstructorStats.detail_reg_inst()

    c = I801C()
    d = I801D()
    # + 4 below because we have the two instances, and each instance has offset base I801B2
    assert ConstructorStats.detail_reg_inst() == n_inst + 4
    b1c = i801b1_c(c)
    assert b1c is c
    b2c = i801b2_c(c)
    assert b2c is c
    b1d = i801b1_d(d)
    assert b1d is d
    b2d = i801b2_d(d)
    assert b2d is d

    assert ConstructorStats.detail_reg_inst() == n_inst + 4  # no extra instances
    del c, b1c, b2c
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    del d, b1d, b2d
    assert ConstructorStats.detail_reg_inst() == n_inst


def test_mi_base_return():
    """Tests returning an offset (non-first MI) base class pointer to a derived instance"""
    from pybind11_tests import (I801B2, I801C, I801D, i801c_b1, i801c_b2, i801d_b1, i801d_b2,
                                i801e_c, i801e_b2)

    n_inst = ConstructorStats.detail_reg_inst()

    c1 = i801c_b1()
    assert type(c1) is I801C
    assert c1.a == 1
    assert c1.b == 2

    d1 = i801d_b1()
    assert type(d1) is I801D
    assert d1.a == 1
    assert d1.b == 2

    assert ConstructorStats.detail_reg_inst() == n_inst + 4

    c2 = i801c_b2()
    assert type(c2) is I801C
    assert c2.a == 1
    assert c2.b == 2

    d2 = i801d_b2()
    assert type(d2) is I801D
    assert d2.a == 1
    assert d2.b == 2

    assert ConstructorStats.detail_reg_inst() == n_inst + 8

    del c2
    assert ConstructorStats.detail_reg_inst() == n_inst + 6
    del c1, d1, d2
    assert ConstructorStats.detail_reg_inst() == n_inst

    # Returning an unregistered derived type with a registered base; we won't
    # pick up the derived type, obviously, but should still work (as an object
    # of whatever type was returned).
    e1 = i801e_c()
    assert type(e1) is I801C
    assert e1.a == 1
    assert e1.b == 2

    e2 = i801e_b2()
    assert type(e2) is I801B2
    assert e2.b == 2
