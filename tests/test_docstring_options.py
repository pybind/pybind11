

def test_docstring_options():
    from pybind11_tests import (test_function1, test_function2, test_function3,
                                test_function4, test_function5, test_function6,
                                test_function7, DocstringTestFoo,
                                test_overloaded1, test_overloaded2, test_overloaded3)

    # options.disable_function_signatures()
    assert not test_function1.__doc__

    assert test_function2.__doc__ == "A custom docstring"

    # docstring specified on just the first overload definition:
    assert test_overloaded1.__doc__ == "Overload docstring"

    # docstring on both overloads:
    assert test_overloaded2.__doc__ == "overload docstring 1\noverload docstring 2"

    # docstring on only second overload:
    assert test_overloaded3.__doc__ == "Overload docstr"

    # options.enable_function_signatures()
    assert test_function3.__doc__ .startswith("test_function3(a: int, b: int) -> None")

    assert test_function4.__doc__ .startswith("test_function4(a: int, b: int) -> None")
    assert test_function4.__doc__ .endswith("A custom docstring\n")

    # options.disable_function_signatures()
    # options.disable_user_defined_docstrings()
    assert not test_function5.__doc__

    # nested options.enable_user_defined_docstrings()
    assert test_function6.__doc__ == "A custom docstring"

    # RAII destructor
    assert test_function7.__doc__ .startswith("test_function7(a: int, b: int) -> None")
    assert test_function7.__doc__ .endswith("A custom docstring\n")

    # Suppression of user-defined docstrings for non-function objects
    assert not DocstringTestFoo.__doc__
    assert not DocstringTestFoo.value_prop.__doc__
