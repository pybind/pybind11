import pytest


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
