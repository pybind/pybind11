
def test_nested_modules():
    import pybind11_tests
    from pybind11_tests.submodule import submodule_func

    assert pybind11_tests.__name__ == "pybind11_tests"
    assert pybind11_tests.submodule.__name__ == "pybind11_tests.submodule"

    assert submodule_func() == "submodule_func()"


def test_reference_internal():
    from pybind11_tests import ConstructorStats
    from pybind11_tests.submodule import A, B

    b = B()
    assert str(b.get_a1()) == "A[1]"
    assert str(b.a1) == "A[1]"
    assert str(b.get_a2()) == "A[2]"
    assert str(b.a2) == "A[2]"

    b.a1 = A(42)
    b.a2 = A(43)
    assert str(b.get_a1()) == "A[42]"
    assert str(b.a1) == "A[42]"
    assert str(b.get_a2()) == "A[43]"
    assert str(b.a2) == "A[43]"

    astats, bstats = ConstructorStats.get(A), ConstructorStats.get(B)
    assert astats.alive() == 2
    assert bstats.alive() == 1
    del b
    assert astats.alive() == 0
    assert bstats.alive() == 0
    assert astats.values() == ['1', '2', '42', '43']
    assert bstats.values() == []
    assert astats.default_constructions == 0
    assert bstats.default_constructions == 1
    assert astats.copy_constructions == 0
    assert bstats.copy_constructions == 0
    # assert astats.move_constructions >= 0  # Don't invoke any
    # assert bstats.move_constructions >= 0  # Don't invoke any
    assert astats.copy_assignments == 2
    assert bstats.copy_assignments == 0
    assert astats.move_assignments == 0
    assert bstats.move_assignments == 0


def test_importing():
    from pybind11_tests import OD
    from collections import OrderedDict

    assert OD is OrderedDict
    assert str(OD([(1, 'a'), (2, 'b')])) == "OrderedDict([(1, 'a'), (2, 'b')])"
