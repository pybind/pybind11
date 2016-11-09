from pybind11_tests import ConstructorStats

def test_docstring_options(capture):
    from pybind11_tests import (test_function1, test_function2, test_function3,
                                test_function4, test_function5, test_function6,
                                test_function7, DocstringTestFoo)

    # docstrings.disable_signatures()
    assert not test_function1.__doc__

    assert test_function2.__doc__ == "A custom docstring"

    # docstrings.enable_signatures()
    assert test_function3.__doc__ .startswith("test_function3(a: int, b: int) -> None")

    assert test_function4.__doc__ .startswith("test_function4(a: int, b: int) -> None")
    assert test_function4.__doc__ .endswith("A custom docstring\n")

    # docstrings.disable_signatures()
    # docstrings.disable_user_defined()
    assert not test_function5.__doc__

    # nested docstrings.enable_user_defined()
    assert test_function6.__doc__ == "A custom docstring"

    # RAII destructor
    assert test_function7.__doc__ .startswith("test_function7(a: int, b: int) -> None")
    assert test_function7.__doc__ .endswith("A custom docstring\n")

    # Suppression of user-defined docstrings for non-function objects
    assert not DocstringTestFoo.__doc__
    assert not DocstringTestFoo.value_prop.__doc__
