import pytest
from pybind11_tests import ExampleMandA, ConstructorStats


def test_methods_and_attributes():
    instance1 = ExampleMandA()
    instance2 = ExampleMandA(32)

    instance1.add1(instance2)
    instance1.add2(instance2)
    instance1.add3(instance2)
    instance1.add4(instance2)
    instance1.add5(instance2)
    instance1.add6(32)
    instance1.add7(32)
    instance1.add8(32)
    instance1.add9(32)
    instance1.add10(32)

    assert str(instance1) == "ExampleMandA[value=320]"
    assert str(instance2) == "ExampleMandA[value=32]"
    assert str(instance1.self1()) == "ExampleMandA[value=320]"
    assert str(instance1.self2()) == "ExampleMandA[value=320]"
    assert str(instance1.self3()) == "ExampleMandA[value=320]"
    assert str(instance1.self4()) == "ExampleMandA[value=320]"
    assert str(instance1.self5()) == "ExampleMandA[value=320]"

    assert instance1.internal1() == 320
    assert instance1.internal2() == 320
    assert instance1.internal3() == 320
    assert instance1.internal4() == 320
    assert instance1.internal5() == 320

    assert instance1.overloaded(1, 1.0) == "(int, float)"
    assert instance1.overloaded(2.0, 2) == "(float, int)"
    assert instance1.overloaded_const(3, 3.0) == "(int, float) const"
    assert instance1.overloaded_const(4.0, 4) == "(float, int) const"

    assert instance1.value == 320
    instance1.value = 100
    assert str(instance1) == "ExampleMandA[value=100]"

    cstats = ConstructorStats.get(ExampleMandA)
    assert cstats.alive() == 2
    del instance1, instance2
    assert cstats.alive() == 0
    assert cstats.values() == ["32"]
    assert cstats.default_constructions == 1
    assert cstats.copy_constructions == 3
    assert cstats.move_constructions >= 1
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0


def test_properties():
    from pybind11_tests import TestProperties

    instance = TestProperties()

    assert instance.def_readonly == 1
    with pytest.raises(AttributeError):
        instance.def_readonly = 2

    instance.def_readwrite = 2
    assert instance.def_readwrite == 2

    assert instance.def_property_readonly == 2
    with pytest.raises(AttributeError):
        instance.def_property_readonly = 3

    instance.def_property = 3
    assert instance.def_property == 3


def test_static_properties():
    from pybind11_tests import TestProperties as Type

    assert Type.def_readonly_static == 1
    with pytest.raises(AttributeError):
        Type.def_readonly_static = 2

    Type.def_readwrite_static = 2
    assert Type.def_readwrite_static == 2

    assert Type.def_property_readonly_static == 2
    with pytest.raises(AttributeError):
        Type.def_property_readonly_static = 3

    Type.def_property_static = 3
    assert Type.def_property_static == 3


@pytest.mark.parametrize("access", ["ro", "rw", "static_ro", "static_rw"])
def test_property_return_value_policies(access):
    from pybind11_tests import TestPropRVP

    if not access.startswith("static"):
        obj = TestPropRVP()
    else:
        obj = TestPropRVP

    ref = getattr(obj, access + "_ref")
    assert ref.value == 1
    ref.value = 2
    assert getattr(obj, access + "_ref").value == 2
    ref.value = 1  # restore original value for static properties

    copy = getattr(obj, access + "_copy")
    assert copy.value == 1
    copy.value = 2
    assert getattr(obj, access + "_copy").value == 1

    copy = getattr(obj, access + "_func")
    assert copy.value == 1
    copy.value = 2
    assert getattr(obj, access + "_func").value == 1


def test_property_rvalue_policy():
    """When returning an rvalue, the return value policy is automatically changed from
    `reference(_internal)` to `move`. The following would not work otherwise.
    """
    from pybind11_tests import TestPropRVP

    instance = TestPropRVP()
    o = instance.rvalue
    assert o.value == 1


def test_property_rvalue_policy_static():
    """When returning an rvalue, the return value policy is automatically changed from
    `reference(_internal)` to `move`. The following would not work otherwise.
    """
    from pybind11_tests import TestPropRVP
    o = TestPropRVP.static_rvalue
    assert o.value == 1


# https://bitbucket.org/pypy/pypy/issues/2447
@pytest.unsupported_on_pypy
def test_dynamic_attributes():
    from pybind11_tests import DynamicClass, CppDerivedDynamicClass

    instance = DynamicClass()
    assert not hasattr(instance, "foo")
    assert "foo" not in dir(instance)

    # Dynamically add attribute
    instance.foo = 42
    assert hasattr(instance, "foo")
    assert instance.foo == 42
    assert "foo" in dir(instance)

    # __dict__ should be accessible and replaceable
    assert "foo" in instance.__dict__
    instance.__dict__ = {"bar": True}
    assert not hasattr(instance, "foo")
    assert hasattr(instance, "bar")

    with pytest.raises(TypeError) as excinfo:
        instance.__dict__ = []
    assert str(excinfo.value) == "__dict__ must be set to a dictionary, not a 'list'"

    cstats = ConstructorStats.get(DynamicClass)
    assert cstats.alive() == 1
    del instance
    assert cstats.alive() == 0

    # Derived classes should work as well
    class PythonDerivedDynamicClass(DynamicClass):
        pass

    for cls in CppDerivedDynamicClass, PythonDerivedDynamicClass:
        derived = cls()
        derived.foobar = 100
        assert derived.foobar == 100

        assert cstats.alive() == 1
        del derived
        assert cstats.alive() == 0


# https://bitbucket.org/pypy/pypy/issues/2447
@pytest.unsupported_on_pypy
def test_cyclic_gc():
    from pybind11_tests import DynamicClass

    # One object references itself
    instance = DynamicClass()
    instance.circular_reference = instance

    cstats = ConstructorStats.get(DynamicClass)
    assert cstats.alive() == 1
    del instance
    assert cstats.alive() == 0

    # Two object reference each other
    i1 = DynamicClass()
    i2 = DynamicClass()
    i1.cycle = i2
    i2.cycle = i1

    assert cstats.alive() == 2
    del i1, i2
    assert cstats.alive() == 0
