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
    assert instance1.overloaded(3,   3) == "(int, int)"
    assert instance1.overloaded(4., 4.) == "(float, float)"
    assert instance1.overloaded_const(5, 5.0) == "(int, float) const"
    assert instance1.overloaded_const(6.0, 6) == "(float, int) const"
    assert instance1.overloaded_const(7,   7) == "(int, int) const"
    assert instance1.overloaded_const(8., 8.) == "(float, float) const"
    assert instance1.overloaded_float(1, 1) == "(float, float)"
    assert instance1.overloaded_float(1, 1.) == "(float, float)"
    assert instance1.overloaded_float(1., 1) == "(float, float)"
    assert instance1.overloaded_float(1., 1.) == "(float, float)"

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


def test_copy_method():
    """Issue #443: calling copied methods fails in Python 3"""
    from pybind11_tests import ExampleMandA

    ExampleMandA.add2c = ExampleMandA.add2
    ExampleMandA.add2d = ExampleMandA.add2b
    a = ExampleMandA(123)
    assert a.value == 123
    a.add2(ExampleMandA(-100))
    assert a.value == 23
    a.add2b(ExampleMandA(20))
    assert a.value == 43
    a.add2c(ExampleMandA(6))
    assert a.value == 49
    a.add2d(ExampleMandA(-7))
    assert a.value == 42


def test_static_properties():
    from pybind11_tests import TestProperties as Type

    assert Type.def_readonly_static == 1
    with pytest.raises(AttributeError) as excinfo:
        Type.def_readonly_static = 2
    assert "can't set attribute" in str(excinfo)

    Type.def_readwrite_static = 2
    assert Type.def_readwrite_static == 2

    assert Type.def_property_readonly_static == 2
    with pytest.raises(AttributeError) as excinfo:
        Type.def_property_readonly_static = 3
    assert "can't set attribute" in str(excinfo)

    Type.def_property_static = 3
    assert Type.def_property_static == 3

    # Static property read and write via instance
    instance = Type()

    Type.def_readwrite_static = 0
    assert Type.def_readwrite_static == 0
    assert instance.def_readwrite_static == 0

    instance.def_readwrite_static = 2
    assert Type.def_readwrite_static == 2
    assert instance.def_readwrite_static == 2

    # It should be possible to override properties in derived classes
    from pybind11_tests import TestPropertiesOverride as TypeOverride

    assert TypeOverride().def_readonly == 99
    assert TypeOverride.def_readonly_static == 99


def test_static_cls():
    """Static property getter and setters expect the type object as the their only argument"""
    from pybind11_tests import TestProperties as Type

    instance = Type()
    assert Type.static_cls is Type
    assert instance.static_cls is Type

    def check_self(self):
        assert self is Type

    Type.static_cls = check_self
    instance.static_cls = check_self


def test_metaclass_override():
    """Overriding pybind11's default metaclass changes the behavior of `static_property`"""
    from pybind11_tests import MetaclassOverride

    assert type(ExampleMandA).__name__ == "pybind11_type"
    assert type(MetaclassOverride).__name__ == "type"

    assert MetaclassOverride.readonly == 1
    assert type(MetaclassOverride.__dict__["readonly"]).__name__ == "pybind11_static_property"

    # Regular `type` replaces the property instead of calling `__set__()`
    MetaclassOverride.readonly = 2
    assert MetaclassOverride.readonly == 2
    assert isinstance(MetaclassOverride.__dict__["readonly"], int)


def test_no_mixed_overloads():
    from pybind11_tests import debug_enabled

    with pytest.raises(RuntimeError) as excinfo:
        ExampleMandA.add_mixed_overloads1()
    assert (str(excinfo.value) ==
            "overloading a method with both static and instance methods is not supported; " +
            ("compile in debug mode for more details" if not debug_enabled else
             "error while attempting to bind static method ExampleMandA.overload_mixed1"
             "() -> str")
            )

    with pytest.raises(RuntimeError) as excinfo:
        ExampleMandA.add_mixed_overloads2()
    assert (str(excinfo.value) ==
            "overloading a method with both static and instance methods is not supported; " +
            ("compile in debug mode for more details" if not debug_enabled else
             "error while attempting to bind instance method ExampleMandA.overload_mixed2"
             "(self: pybind11_tests.ExampleMandA, arg0: int, arg1: int) -> str")
            )


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


def test_noconvert_args(msg):
    import pybind11_tests as m

    a = m.ArgInspector()
    assert msg(a.f("hi")) == """
        loading ArgInspector1 argument WITH conversion allowed.  Argument value = hi
    """
    assert msg(a.g("this is a", "this is b")) == """
        loading ArgInspector1 argument WITHOUT conversion allowed.  Argument value = this is a
        loading ArgInspector1 argument WITH conversion allowed.  Argument value = this is b
        13
        loading ArgInspector2 argument WITH conversion allowed.  Argument value = (default arg inspector 2)
    """  # noqa: E501 line too long
    assert msg(a.g("this is a", "this is b", 42)) == """
        loading ArgInspector1 argument WITHOUT conversion allowed.  Argument value = this is a
        loading ArgInspector1 argument WITH conversion allowed.  Argument value = this is b
        42
        loading ArgInspector2 argument WITH conversion allowed.  Argument value = (default arg inspector 2)
    """  # noqa: E501 line too long
    assert msg(a.g("this is a", "this is b", 42, "this is d")) == """
        loading ArgInspector1 argument WITHOUT conversion allowed.  Argument value = this is a
        loading ArgInspector1 argument WITH conversion allowed.  Argument value = this is b
        42
        loading ArgInspector2 argument WITH conversion allowed.  Argument value = this is d
    """
    assert (a.h("arg 1") ==
            "loading ArgInspector2 argument WITHOUT conversion allowed.  Argument value = arg 1")
    assert msg(m.arg_inspect_func("A1", "A2")) == """
        loading ArgInspector2 argument WITH conversion allowed.  Argument value = A1
        loading ArgInspector1 argument WITHOUT conversion allowed.  Argument value = A2
    """

    assert m.floats_preferred(4) == 2.0
    assert m.floats_only(4.0) == 2.0
    with pytest.raises(TypeError) as excinfo:
        m.floats_only(4)
    assert msg(excinfo.value) == """
        floats_only(): incompatible function arguments. The following argument types are supported:
            1. (f: float) -> float

        Invoked with: 4
    """

    assert m.ints_preferred(4) == 2
    assert m.ints_preferred(True) == 0
    with pytest.raises(TypeError) as excinfo:
        m.ints_preferred(4.0)
    assert msg(excinfo.value) == """
        ints_preferred(): incompatible function arguments. The following argument types are supported:
            1. (i: int) -> int

        Invoked with: 4.0
    """  # noqa: E501 line too long

    assert m.ints_only(4) == 2
    with pytest.raises(TypeError) as excinfo:
        m.ints_only(4.0)
    assert msg(excinfo.value) == """
        ints_only(): incompatible function arguments. The following argument types are supported:
            1. (i: int) -> int

        Invoked with: 4.0
    """


def test_bad_arg_default(msg):
    from pybind11_tests import debug_enabled, bad_arg_def_named, bad_arg_def_unnamed

    with pytest.raises(RuntimeError) as excinfo:
        bad_arg_def_named()
    assert msg(excinfo.value) == (
        "arg(): could not convert default argument 'a: NotRegistered' in function 'should_fail' "
        "into a Python object (type not registered yet?)"
        if debug_enabled else
        "arg(): could not convert default argument into a Python object (type not registered "
        "yet?). Compile in debug mode for more information."
    )

    with pytest.raises(RuntimeError) as excinfo:
        bad_arg_def_unnamed()
    assert msg(excinfo.value) == (
        "arg(): could not convert default argument 'NotRegistered' in function 'should_fail' "
        "into a Python object (type not registered yet?)"
        if debug_enabled else
        "arg(): could not convert default argument into a Python object (type not registered "
        "yet?). Compile in debug mode for more information."
    )


def test_accepts_none(msg):
    from pybind11_tests import (NoneTester,
                                no_none1, no_none2, no_none3, no_none4, no_none5,
                                ok_none1, ok_none2, ok_none3, ok_none4, ok_none5)

    a = NoneTester()
    assert no_none1(a) == 42
    assert no_none2(a) == 42
    assert no_none3(a) == 42
    assert no_none4(a) == 42
    assert no_none5(a) == 42
    assert ok_none1(a) == 42
    assert ok_none2(a) == 42
    assert ok_none3(a) == 42
    assert ok_none4(a) == 42
    assert ok_none5(a) == 42

    with pytest.raises(TypeError) as excinfo:
        no_none1(None)
    assert "incompatible function arguments" in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        no_none2(None)
    assert "incompatible function arguments" in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        no_none3(None)
    assert "incompatible function arguments" in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        no_none4(None)
    assert "incompatible function arguments" in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        no_none5(None)
    assert "incompatible function arguments" in str(excinfo.value)

    # The first one still raises because you can't pass None as a lvalue reference arg:
    with pytest.raises(TypeError) as excinfo:
        assert ok_none1(None) == -1
    assert msg(excinfo.value) == """
        ok_none1(): incompatible function arguments. The following argument types are supported:
            1. (arg0: m.NoneTester) -> int

        Invoked with: None
    """

    # The rest take the argument as pointer or holder, and accept None:
    assert ok_none2(None) == -1
    assert ok_none3(None) == -1
    assert ok_none4(None) == -1
    assert ok_none5(None) == -1


def test_str_issue(msg):
    """#283: __str__ called on uninitialized instance when constructor arguments invalid"""
    from pybind11_tests import StrIssue

    assert str(StrIssue(3)) == "StrIssue[3]"

    with pytest.raises(TypeError) as excinfo:
        str(StrIssue("no", "such", "constructor"))
    assert msg(excinfo.value) == """
        __init__(): incompatible constructor arguments. The following argument types are supported:
            1. m.StrIssue(arg0: int)
            2. m.StrIssue()

        Invoked with: 'no', 'such', 'constructor'
    """
