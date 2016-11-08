from pybind11_tests import ConstructorStats

def test_docstring_options(capture):
    from pybind11_tests import (test_function1, test_function2, test_function3, test_function4)

    assert not test_function1.__doc__
    assert test_function2.__doc__ == "A custom docstring"
    assert test_function3.__doc__ .startswith("test_function3(a: int, b: int) -> None")
    assert test_function4.__doc__ .startswith("test_function4(a: int, b: int) -> None")
    assert test_function4.__doc__ .endswith("A custom docstring\n")
