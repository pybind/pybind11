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


def test_multiple_inheritance_python():
    from pybind11_tests import Base1, Base2

    class MI1(Base1, Base2):
        def __init__(self, i, j):
            Base1.__init__(self, i)
            Base2.__init__(self, j)

    class B1(object):
        def v(self):
            return 1

    class MI2(B1, Base1, Base2):
        def __init__(self, i, j):
            B1.__init__(self)
            Base1.__init__(self, i)
            Base2.__init__(self, j)

    class MI3(MI2):
        def __init__(self, i, j):
            MI2.__init__(self, i, j)

    class MI4(MI3, Base2):
        def __init__(self, i, j, k):
            MI2.__init__(self, j, k)
            Base2.__init__(self, i)

    class MI5(Base2, B1, Base1):
        def __init__(self, i, j):
            B1.__init__(self)
            Base1.__init__(self, i)
            Base2.__init__(self, j)

    class MI6(Base2, B1):
        def __init__(self, i):
            Base2.__init__(self, i)
            B1.__init__(self)

    class B2(B1):
        def v(self):
            return 2

    class B3(object):
        def v(self):
            return 3

    class B4(B3, B2):
        def v(self):
            return 4

    class MI7(B4, MI6):
        def __init__(self, i):
            B4.__init__(self)
            MI6.__init__(self, i)

    class MI8(MI6, B3):
        def __init__(self, i):
            MI6.__init__(self, i)
            B3.__init__(self)

    class MI8b(B3, MI6):
        def __init__(self, i):
            B3.__init__(self)
            MI6.__init__(self, i)

    mi1 = MI1(1, 2)
    assert mi1.foo() == 1
    assert mi1.bar() == 2

    mi2 = MI2(3, 4)
    assert mi2.v() == 1
    assert mi2.foo() == 3
    assert mi2.bar() == 4

    mi3 = MI3(5, 6)
    assert mi3.v() == 1
    assert mi3.foo() == 5
    assert mi3.bar() == 6

    mi4 = MI4(7, 8, 9)
    assert mi4.v() == 1
    assert mi4.foo() == 8
    assert mi4.bar() == 7

    mi5 = MI5(10, 11)
    assert mi5.v() == 1
    assert mi5.foo() == 10
    assert mi5.bar() == 11

    mi6 = MI6(12)
    assert mi6.v() == 1
    assert mi6.bar() == 12

    mi7 = MI7(13)
    assert mi7.v() == 4
    assert mi7.bar() == 13

    mi8 = MI8(14)
    assert mi8.v() == 1
    assert mi8.bar() == 14

    mi8b = MI8b(15)
    assert mi8b.v() == 3
    assert mi8b.bar() == 15


def test_multiple_inheritance_python_many_bases():
    from pybind11_tests import (BaseN1,  BaseN2,  BaseN3,  BaseN4,  BaseN5,  BaseN6,  BaseN7,
                                BaseN8,  BaseN9,  BaseN10, BaseN11, BaseN12, BaseN13, BaseN14,
                                BaseN15, BaseN16, BaseN17)

    class MIMany14(BaseN1, BaseN2, BaseN3, BaseN4):
        def __init__(self):
            BaseN1.__init__(self, 1)
            BaseN2.__init__(self, 2)
            BaseN3.__init__(self, 3)
            BaseN4.__init__(self, 4)

    class MIMany58(BaseN5, BaseN6, BaseN7, BaseN8):
        def __init__(self):
            BaseN5.__init__(self, 5)
            BaseN6.__init__(self, 6)
            BaseN7.__init__(self, 7)
            BaseN8.__init__(self, 8)

    class MIMany916(BaseN9, BaseN10, BaseN11, BaseN12, BaseN13, BaseN14, BaseN15, BaseN16):
        def __init__(self):
            BaseN9.__init__(self, 9)
            BaseN10.__init__(self, 10)
            BaseN11.__init__(self, 11)
            BaseN12.__init__(self, 12)
            BaseN13.__init__(self, 13)
            BaseN14.__init__(self, 14)
            BaseN15.__init__(self, 15)
            BaseN16.__init__(self, 16)

    class MIMany19(MIMany14, MIMany58, BaseN9):
        def __init__(self):
            MIMany14.__init__(self)
            MIMany58.__init__(self)
            BaseN9.__init__(self, 9)

    class MIMany117(MIMany14, MIMany58, MIMany916, BaseN17):
        def __init__(self):
            MIMany14.__init__(self)
            MIMany58.__init__(self)
            MIMany916.__init__(self)
            BaseN17.__init__(self, 17)

    # Inherits from 4 registered C++ classes: can fit in one pointer on any modern arch:
    a = MIMany14()
    for i in range(1, 4):
        assert getattr(a, "f" + str(i))() == 2 * i

    # Inherits from 8: requires 1/2 pointers worth of holder flags on 32/64-bit arch:
    b = MIMany916()
    for i in range(9, 16):
        assert getattr(b, "f" + str(i))() == 2 * i

    # Inherits from 9: requires >= 2 pointers worth of holder flags
    c = MIMany19()
    for i in range(1, 9):
        assert getattr(c, "f" + str(i))() == 2 * i

    # Inherits from 17: requires >= 3 pointers worth of holder flags
    d = MIMany117()
    for i in range(1, 17):
        assert getattr(d, "f" + str(i))() == 2 * i


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
