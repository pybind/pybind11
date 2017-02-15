import pytest


def test_inheritance(msg):
    from pybind11_tests import Pet, Dog, Rabbit, Hamster, Chimera, dog_bark, pet_name_species

    roger = Rabbit('Rabbit')
    assert roger.name() + " is a " + roger.species() == "Rabbit is a parrot"
    assert pet_name_species(roger) == "Rabbit is a parrot"

    polly = Pet('Polly', 'parrot')
    assert polly.name() + " is a " + polly.species() == "Polly is a parrot"
    assert pet_name_species(polly) == "Polly is a parrot"

    molly = Dog('Molly')
    assert molly.name() + " is a " + molly.species() == "Molly is a dog"
    assert pet_name_species(molly) == "Molly is a dog"

    fred = Hamster('Fred')
    assert fred.name() + " is a " + fred.species() == "Fred is a rodent"

    assert dog_bark(molly) == "Woof!"

    with pytest.raises(TypeError) as excinfo:
        dog_bark(polly)
    assert msg(excinfo.value) == """
        dog_bark(): incompatible function arguments. The following argument types are supported:
            1. (arg0: m.Dog) -> str

        Invoked with: <m.Pet object at 0>
    """

    with pytest.raises(TypeError) as excinfo:
        Chimera("lion", "goat")
    assert "No constructor defined!" in str(excinfo.value)


def test_automatic_upcasting():
    from pybind11_tests import return_class_1, return_class_2, return_class_n, return_none

    assert type(return_class_1()).__name__ == "DerivedClass1"
    assert type(return_class_2()).__name__ == "DerivedClass2"
    assert type(return_none()).__name__ == "NoneType"
    # Repeat these a few times in a random order to ensure no invalid caching
    # is applied
    assert type(return_class_n(1)).__name__ == "DerivedClass1"
    assert type(return_class_n(2)).__name__ == "DerivedClass2"
    assert type(return_class_n(0)).__name__ == "BaseClass"
    assert type(return_class_n(2)).__name__ == "DerivedClass2"
    assert type(return_class_n(2)).__name__ == "DerivedClass2"
    assert type(return_class_n(0)).__name__ == "BaseClass"
    assert type(return_class_n(1)).__name__ == "DerivedClass1"


def test_isinstance():
    from pybind11_tests import test_isinstance, Pet, Dog

    objects = [tuple(), dict(), Pet("Polly", "parrot")] + [Dog("Molly")] * 4
    expected = (True, True, True, True, True, False, False)
    assert test_isinstance(objects) == expected


def test_holder():
    from pybind11_tests import test_mismatched_holder_type_1, test_mismatched_holder_type_2

    with pytest.raises(RuntimeError) as excinfo:
        test_mismatched_holder_type_1()

    assert str(excinfo.value) == ("generic_type: type \"MismatchDerived1\" does not have "
                                  "a non-default holder type while its base "
                                  "\"MismatchBase1\" does")

    with pytest.raises(RuntimeError) as excinfo:
        test_mismatched_holder_type_2()

    assert str(excinfo.value) == ("generic_type: type \"MismatchDerived2\" has a "
                                  "non-default holder type while its base "
                                  "\"MismatchBase2\" does not")
