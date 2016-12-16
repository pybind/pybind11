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


def test_multiple_inheritance_virtbase():
    from pybind11_tests import Base12a, bar_base2a, bar_base2a_sharedptr

    class MITypePy(Base12a):
        def __init__(self, i, j):
            Base12a.__init__(self, i, j)

    mt = MITypePy(3, 4)
    assert mt.bar() == 4
    assert bar_base2a(mt) == 4
    assert bar_base2a_sharedptr(mt) == 4
