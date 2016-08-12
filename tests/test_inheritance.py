import pytest


def test_inheritance(msg):
    from pybind11_tests import Pet, Dog, Rabbit, dog_bark, pet_name_species

    roger = Rabbit('Rabbit')
    assert roger.name() + " is a " + roger.species() == "Rabbit is a parrot"
    assert pet_name_species(roger) == "Rabbit is a parrot"

    polly = Pet('Polly', 'parrot')
    assert polly.name() + " is a " + polly.species() == "Polly is a parrot"
    assert pet_name_species(polly) == "Polly is a parrot"

    molly = Dog('Molly')
    assert molly.name() + " is a " + molly.species() == "Molly is a dog"
    assert pet_name_species(molly) == "Molly is a dog"

    assert dog_bark(molly) == "Woof!"

    with pytest.raises(TypeError) as excinfo:
        dog_bark(polly)
    assert msg(excinfo.value) == """
        Incompatible function arguments. The following argument types are supported:
            1. (arg0: m.Dog) -> str
            Invoked with: <m.Pet object at 0>
    """


def test_automatic_upcasting():
    from pybind11_tests import return_class_1, return_class_2, return_none

    assert type(return_class_1()).__name__ == "DerivedClass1"
    assert type(return_class_2()).__name__ == "DerivedClass2"
    assert type(return_none()).__name__ == "NoneType"
