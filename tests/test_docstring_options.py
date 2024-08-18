from __future__ import annotations

from pybind11_tests import docstring_options as m


def test_docstring_options():
    # options.disable_function_signatures()
    assert not m.test_function1.__doc__

    assert m.test_function2.__doc__ == "A custom docstring"

    # docstring specified on just the first overload definition:
    assert m.test_overloaded1.__doc__ == (
        "test_overloaded1(i: int) -> None\n"
        "test_overloaded1(d: float) -> None\n"
        "Overload docstring"
    )

    # docstring on both overloads:
    assert m.test_overloaded2.__doc__ == (
        "test_overloaded2(i: int) -> None\n"
        "test_overloaded2(d: float) -> None\n"
        "overload docstring 1\n"
        "overload docstring 2"
    )

    # docstring on only second overload:
    assert m.test_overloaded3.__doc__ == (
        "test_overloaded3(i: int) -> None\n"
        "test_overloaded3(d: float) -> None\n"
        "Overload docstr"
    )

    # Check overload configuration behaviour matches the documentation
    assert m.test_overloaded4.__doc__ == (
        "test_overloaded4(arg0: int, arg1: int) -> int\n"
        "test_overloaded4(arg0: float, arg1: float) -> float\n"
        "test_overloaded4(arg0: None, arg1: None) -> None\n"
        "A function which adds two numbers.\n\n"
        "Internally, a simple addition is performed.\n"
        "Both numbers can be None, and None will be returned."
    )

    assert m.test_overloaded5.__doc__ == (
        "test_overloaded5(arg0: int, arg1: int) -> int\n"
        "test_overloaded5(arg0: float, arg1: float) -> float\n"
        "Overloaded function.\n"
        "\n"
        "1. test_overloaded5(arg0: int, arg1: int) -> int\n"
        "\n"
        "Add two integers together.\n"
        "\n"
        "2. test_overloaded5(arg0: float, arg1: float) -> float\n"
        "\n"
        "Add two floating point numbers together.\n"
    )

    # options.enable_function_signatures()
    assert m.test_function3.__doc__.startswith("test_function3(a: int, b: int) -> None")

    assert m.test_function4.__doc__.startswith("test_function4(a: int, b: int) -> None")
    assert m.test_function4.__doc__.endswith("A custom docstring\n")

    # options.disable_function_signatures()
    # options.disable_user_defined_docstrings()
    assert not m.test_function5.__doc__

    # nested options.enable_user_defined_docstrings()
    assert m.test_function6.__doc__ == "A custom docstring"

    # RAII destructor
    assert m.test_function7.__doc__.startswith("test_function7(a: int, b: int) -> None")
    assert m.test_function7.__doc__.endswith("A custom docstring\n")

    # when all options are disabled, no docstring (instead of an empty one) should be generated
    assert m.test_function8.__doc__ is None

    # Suppression of user-defined docstrings for non-function objects
    assert not m.DocstringTestFoo.__doc__
    assert not m.DocstringTestFoo.value_prop.__doc__

    # Check existig behaviour of enum docstings
    assert (
        m.DocstringTestEnum1.__doc__
        == "Enum docstring\n\nMembers:\n\n  Member1\n\n  Member2"
    )

    # options.enable_enum_members_docstring()
    assert (
        m.DocstringTestEnum2.__doc__
        == "Enum docstring\n\nMembers:\n\n  Member1\n\n  Member2"
    )

    # options.disable_enum_members_docstring()
    assert m.DocstringTestEnum3.__doc__ == "Enum docstring"

    # options.disable_user_defined_docstrings()
    assert m.DocstringTestEnum4.__doc__ == "Members:\n\n  Member1\n\n  Member2"

    # options.disable_user_defined_docstrings()
    # options.disable_enum_members_docstring()
    # When all options are disabled, no docstring (instead of an empty one) should be generated
    assert m.DocstringTestEnum5.__doc__ is None
