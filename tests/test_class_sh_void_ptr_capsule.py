import pytest

from pybind11_tests import class_sh_void_ptr_capsule as m


class Valid:
    def __init__(self):
        self.capsule_generated = False

    def as_pybind11_tests_class_sh_void_ptr_capsule_Valid(self):
        self.capsule_generated = True
        return m.create_test_void_ptr_capsule()


class NoConversion:
    def __init__(self):
        self.capsule_generated = False


class NoCapsuleReturned:
    def __init__(self):
        self.capsule_generated = False

    def as_pybind11_tests_class_sh_void_ptr_capsule_NoCapsuleReturned(
        self,
    ):
        pass


class AsAnotherObject:
    def __init__(self):
        self.capsule_generated = False

    def as_pybind11_tests_class_sh_void_ptr_capsule_Valid(self):
        self.capsule_generated = True
        return m.create_test_void_ptr_capsule()


@pytest.mark.parametrize(
    ("ctor", "caller", "expected"),
    [
        (Valid, m.get_from_valid_capsule, 1),
        (AsAnotherObject, m.get_from_valid_capsule, 1),
    ],
)
def test_valid_as_void_ptr_capsule_function(ctor, caller, expected):
    obj = ctor()
    assert caller(obj) == expected
    assert obj.capsule_generated


@pytest.mark.parametrize(
    ("ctor", "caller"),
    [
        (NoConversion, m.get_from_no_conversion_capsule),
        (NoCapsuleReturned, m.get_from_no_capsule_returned),
    ],
)
def test_invalid_as_void_ptr_capsule_function(ctor, caller):
    obj = ctor()
    with pytest.raises(TypeError):
        caller(obj)
    assert not obj.capsule_generated


@pytest.mark.parametrize(
    ("ctor", "caller", "pointer_type", "capsule_generated"),
    [
        (AsAnotherObject, m.get_from_shared_ptr_valid_capsule, "shared_ptr", True),
        (AsAnotherObject, m.get_from_unique_ptr_valid_capsule, "unique_ptr", True),
    ],
)
def test_as_void_ptr_capsule_unsupported(ctor, caller, pointer_type, capsule_generated):
    obj = ctor()
    with pytest.raises(RuntimeError) as excinfo:
        caller(obj)
    assert pointer_type in str(excinfo.value)
    assert obj.capsule_generated == capsule_generated


def test_type_with_getattr():
    obj = m.TypeWithGetattr()
    assert obj.get_42() == 42
    assert obj.something == "GetAttr: something"


def test_multiple_inheritance_getattr():
    d1 = m.Derived1()
    assert d1.foo() == 0
    assert d1.bar() == 1
    assert d1.prop1 == "Base GetAttr: prop1"

    d2 = m.Derived2()
    assert d2.foo() == 0
    assert d2.bar() == 2
    assert d2.prop2 == "Base GetAttr: prop2"


def test_pass_unspecified_base():
    assert m.PassUnspecBase(m.UnspecDerived()) == 230


def test_explore():
    obj = m.ExploreType()
    assert obj.plain_mfun() == 42
    assert obj.something == "GetAttr: something"
    ii = obj.isinstance_method
    assert ii == "GetAttr: isinstance_method"
    assert m.is_instance_method(m.ExploreType, "plain_mfun")
    assert not m.is_instance_method(m.ExploreType, "something")
    assert m.is_instance_method(obj, "plain_mfun")
    assert not m.is_instance_method(obj, "something")


def test_type_is_pybind11_class_():
    assert m.type_is_pybind11_class_(m.ExploreType)
    assert not m.type_is_pybind11_class_(Valid)


def test_valid_debug():
    obj = Valid()
    assert m.get_from_valid_capsule(obj) == 1
    assert obj.capsule_generated
