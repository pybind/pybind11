import pytest

from pybind11_tests import class_ as m
from pybind11_tests import UserType, ConstructorStats


def test_repr():
    # In Python 3.3+, repr() accesses __qualname__
    assert "pybind11_type" in repr(type(UserType))
    assert "UserType" in repr(UserType)


def test_instance(msg):
    with pytest.raises(TypeError) as excinfo:
        m.NoConstructor()
    assert msg(excinfo.value) == "m.class_.NoConstructor: No constructor defined!"

    instance = m.NoConstructor.new_instance()

    cstats = ConstructorStats.get(m.NoConstructor)
    assert cstats.alive() == 1
    del instance
    assert cstats.alive() == 0


def test_docstrings(doc):
    assert doc(UserType) == "A `py::class_` type for testing"
    assert UserType.__name__ == "UserType"
    assert UserType.__module__ == "pybind11_tests"
    assert UserType.get_value.__name__ == "get_value"
    assert UserType.get_value.__module__ == "pybind11_tests"

    assert doc(UserType.get_value) == """
        get_value(self: m.UserType) -> int

        Get value using a method
    """
    assert doc(UserType.value) == "Get value using a property"

    assert doc(m.NoConstructor.new_instance) == """
        new_instance() -> m.class_.NoConstructor

        Return an instance
    """


def test_inheritance(msg):
    roger = m.Rabbit('Rabbit')
    assert roger.name() + " is a " + roger.species() == "Rabbit is a parrot"
    assert m.pet_name_species(roger) == "Rabbit is a parrot"

    polly = m.Pet('Polly', 'parrot')
    assert polly.name() + " is a " + polly.species() == "Polly is a parrot"
    assert m.pet_name_species(polly) == "Polly is a parrot"

    molly = m.Dog('Molly')
    assert molly.name() + " is a " + molly.species() == "Molly is a dog"
    assert m.pet_name_species(molly) == "Molly is a dog"

    fred = m.Hamster('Fred')
    assert fred.name() + " is a " + fred.species() == "Fred is a rodent"

    assert m.dog_bark(molly) == "Woof!"

    with pytest.raises(TypeError) as excinfo:
        m.dog_bark(polly)
    assert msg(excinfo.value) == """
        dog_bark(): incompatible function arguments. The following argument types are supported:
            1. (arg0: m.class_.Dog) -> str

        Invoked with: <m.class_.Pet object at 0>
    """

    with pytest.raises(TypeError) as excinfo:
        m.Chimera("lion", "goat")
    assert "No constructor defined!" in str(excinfo.value)


def test_automatic_upcasting():
    assert type(m.return_class_1()).__name__ == "DerivedClass1"
    assert type(m.return_class_2()).__name__ == "DerivedClass2"
    assert type(m.return_none()).__name__ == "NoneType"
    # Repeat these a few times in a random order to ensure no invalid caching is applied
    assert type(m.return_class_n(1)).__name__ == "DerivedClass1"
    assert type(m.return_class_n(2)).__name__ == "DerivedClass2"
    assert type(m.return_class_n(0)).__name__ == "BaseClass"
    assert type(m.return_class_n(2)).__name__ == "DerivedClass2"
    assert type(m.return_class_n(2)).__name__ == "DerivedClass2"
    assert type(m.return_class_n(0)).__name__ == "BaseClass"
    assert type(m.return_class_n(1)).__name__ == "DerivedClass1"


def test_isinstance():
    objects = [tuple(), dict(), m.Pet("Polly", "parrot")] + [m.Dog("Molly")] * 4
    expected = (True, True, True, True, True, False, False)
    assert m.check_instances(objects) == expected


def test_mismatched_holder():
    import re

    with pytest.raises(RuntimeError) as excinfo:
        m.mismatched_holder_1()
    assert re.match('generic_type: type ".*MismatchDerived1" does not have a non-default '
                    'holder type while its base ".*MismatchBase1" does', str(excinfo.value))

    with pytest.raises(RuntimeError) as excinfo:
        m.mismatched_holder_2()
    assert re.match('generic_type: type ".*MismatchDerived2" has a non-default holder type '
                    'while its base ".*MismatchBase2" does not', str(excinfo.value))


def test_override_static():
    """#511: problem with inheritance + overwritten def_static"""
    b = m.MyBase.make()
    d1 = m.MyDerived.make2()
    d2 = m.MyDerived.make()

    assert isinstance(b, m.MyBase)
    assert isinstance(d1, m.MyDerived)
    assert isinstance(d2, m.MyDerived)


def test_implicit_conversion_life_support():
    """Ensure the lifetime of temporary objects created for implicit conversions"""
    assert m.implicitly_convert_argument(UserType(5)) == 5
    assert m.implicitly_convert_variable(UserType(5)) == 5

    assert "outside a bound function" in m.implicitly_convert_variable_fail(UserType(5))
