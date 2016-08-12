import pytest


def test_inheritance(capture, msg):
    from pybind11_tests import Pet, Dog, Rabbit, dog_bark, pet_print

    roger = Rabbit('Rabbit')
    assert roger.name() + " is a " + roger.species() == "Rabbit is a parrot"
    with capture:
        pet_print(roger)
    assert capture == "Rabbit is a parrot"

    polly = Pet('Polly', 'parrot')
    assert polly.name() + " is a " + polly.species() == "Polly is a parrot"
    with capture:
        pet_print(polly)
    assert capture == "Polly is a parrot"

    molly = Dog('Molly')
    assert molly.name() + " is a " + molly.species() == "Molly is a dog"
    with capture:
        pet_print(molly)
    assert capture == "Molly is a dog"

    with capture:
        dog_bark(molly)
    assert capture == "Woof!"

    with pytest.raises(TypeError) as excinfo:
        dog_bark(polly)
    assert msg(excinfo.value) == """
        Incompatible function arguments. The following argument types are supported:
            1. (arg0: m.Dog) -> None
            Invoked with: <m.Pet object at 0>
    """


def test_callbacks(capture):
    from functools import partial
    from pybind11_tests import (test_callback1, test_callback2, test_callback3,
                                test_callback4, test_callback5)

    def func1():
        print('Callback function 1 called!')

    def func2(a, b, c, d):
        print('Callback function 2 called : {}, {}, {}, {}'.format(a, b, c, d))
        return d

    def func3(a):
        print('Callback function 3 called : {}'.format(a))

    with capture:
        assert test_callback1(func1) is False
    assert capture == "Callback function 1 called!"
    with capture:
        assert test_callback2(func2) == 5
    assert capture == "Callback function 2 called : Hello, x, True, 5"
    with capture:
        assert test_callback1(partial(func2, "Hello", "from", "partial", "object")) is False
    assert capture == "Callback function 2 called : Hello, from, partial, object"
    with capture:
        assert test_callback1(partial(func3, "Partial object with one argument")) is False
    assert capture == "Callback function 3 called : Partial object with one argument"
    with capture:
        test_callback3(lambda i: i + 1)
    assert capture == "func(43) = 44"

    f = test_callback4()
    assert f(43) == 44
    f = test_callback5()
    assert f(number=43) == 44


def test_lambda_closure_cleanup():
    from pybind11_tests import test_cleanup, payload_cstats

    test_cleanup()
    cstats = payload_cstats()
    assert cstats.alive() == 0
    assert cstats.copy_constructions == 1
    assert cstats.move_constructions >= 1


def test_cpp_function_roundtrip(capture):
    """Test if passing a function pointer from C++ -> Python -> C++ yields the original pointer"""
    from pybind11_tests import dummy_function, dummy_function2, test_dummy_function, roundtrip

    with capture:
        test_dummy_function(dummy_function)
    assert capture == """
        argument matches dummy_function
        eval(1) = 2
    """
    with capture:
        test_dummy_function(roundtrip(dummy_function))
    assert capture == """
        roundtrip..
        argument matches dummy_function
        eval(1) = 2
    """
    with capture:
        assert roundtrip(None) is None
    assert capture == "roundtrip (got None).."
    with capture:
        test_dummy_function(lambda x: x + 2)
    assert capture == """
        could not convert to a function pointer.
        eval(1) = 3
    """

    with capture:
        with pytest.raises(TypeError) as excinfo:
            test_dummy_function(dummy_function2)
        assert "Incompatible function arguments" in str(excinfo.value)
    assert capture == "could not convert to a function pointer."

    with capture:
        with pytest.raises(TypeError) as excinfo:
            test_dummy_function(lambda x, y: x + y)
        assert any(s in str(excinfo.value) for s in ("missing 1 required positional argument",
                                                     "takes exactly 2 arguments"))
    assert capture == "could not convert to a function pointer."


def test_function_signatures(doc):
    from pybind11_tests import test_callback3, test_callback4

    assert doc(test_callback3) == "test_callback3(arg0: Callable[[int], int]) -> None"
    assert doc(test_callback4) == "test_callback4() -> Callable[[int], int]"
