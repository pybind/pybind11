from __future__ import annotations

import warnings

import pytest

import pybind11_tests  # noqa: F401
from pybind11_tests import warnings_ as m


@pytest.mark.parametrize(
    ("expected_category", "expected_message", "expected_value", "module_function"),
    [
        (Warning, "This is simple warning", 21, m.warn_and_return_value),
        (RuntimeWarning, "This is RuntimeWarning", None, m.warn_with_default_category),
        (FutureWarning, "This is FutureWarning", None, m.warn_with_different_category),
    ],
)
def test_warning_simple(
    expected_category, expected_message, expected_value, module_function
):
    with pytest.warns(Warning) as excinfo:
        value = module_function()

    assert issubclass(excinfo[0].category, expected_category)
    assert str(excinfo[0].message) == expected_message
    assert value == expected_value


def test_warning_wrong_subclass_fail():
    with pytest.raises(Exception) as excinfo:
        m.warn_with_invalid_category()

    assert issubclass(excinfo.type, RuntimeError)
    assert (
        str(excinfo.value)
        == "pybind11::warnings::warn(): cannot raise warning, category must be a subclass of PyExc_Warning!"
    )


def test_warning_double_register_fail():
    with pytest.raises(Exception) as excinfo:
        m.register_duplicate_warning()

    assert issubclass(excinfo.type, RuntimeError)
    assert (
        str(excinfo.value)
        == 'pybind11::warnings::new_warning_type(): an attribute with name "CustomWarning" exists already.'
    )


def test_warning_register():
    assert m.CustomWarning is not None

    with pytest.warns(m.CustomWarning) as excinfo:
        warnings.warn("This is warning from Python!", m.CustomWarning, stacklevel=1)

    assert issubclass(excinfo[0].category, DeprecationWarning)
    assert str(excinfo[0].message) == "This is warning from Python!"


def test_warning_custom():
    with pytest.warns(m.CustomWarning) as excinfo:
        value = m.warn_with_custom_type()

    assert issubclass(excinfo[0].category, DeprecationWarning)
    assert str(excinfo[0].message) == "This is CustomWarning"
    assert value == 37
