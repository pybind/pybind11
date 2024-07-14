from __future__ import annotations

import warnings

import pytest

import pybind11_tests  # noqa: F401
from pybind11_tests import warnings_ as m


@pytest.mark.parametrize(
    ("expected_category", "expected_message", "expected_value", "module_function"),
    [
        (Warning, "Warning was raised!", 21, m.raise_and_return),
        (RuntimeWarning, "RuntimeWarning is raised!", None, m.raise_default),
        (UnicodeWarning, "UnicodeWarning is raised!", None, m.raise_from_cpython),
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


def test_warning_fail():
    with pytest.raises(Exception) as excinfo:
        m.raise_and_fail()

    assert issubclass(excinfo.type, RuntimeError)
    assert (
        str(excinfo.value)
        == "raise_warning(): cannot raise warning, category must be a subclass of PyExc_Warning!"
    )


def test_warning_register():
    assert m.CustomWarning is not None
    assert issubclass(m.CustomWarning, DeprecationWarning)

    with pytest.warns(m.CustomWarning) as excinfo:
        warnings.warn("This is warning from Python!", m.CustomWarning, stacklevel=1)

    assert issubclass(excinfo[0].category, DeprecationWarning)
    assert issubclass(excinfo[0].category, m.CustomWarning)
    assert str(excinfo[0].message) == "This is warning from Python!"


@pytest.mark.parametrize(
    (
        "expected_category",
        "expected_base",
        "expected_message",
        "expected_value",
        "module_function",
    ),
    [
        (
            m.CustomWarning,
            DeprecationWarning,
            "CustomWarning was raised!",
            37,
            m.raise_custom,
        ),
        (
            m.CustomWarning,
            DeprecationWarning,
            "This is raised from a wrapper.",
            42,
            m.raise_with_wrapper,
        ),
    ],
)
def test_warning_custom(
    expected_category, expected_base, expected_message, expected_value, module_function
):
    with pytest.warns(expected_category) as excinfo:
        value = module_function()

    assert issubclass(excinfo[0].category, expected_base)
    assert issubclass(excinfo[0].category, expected_category)
    assert str(excinfo[0].message) == expected_message
    assert value == expected_value


@pytest.mark.parametrize(
    ("expected_category", "module_function"),
    [
        (Warning, m.raise_base_warning),
        (BytesWarning, m.raise_bytes_warning),
        (DeprecationWarning, m.raise_deprecation_warning),
        (FutureWarning, m.raise_future_warning),
        (ImportWarning, m.raise_import_warning),
        (PendingDeprecationWarning, m.raise_pending_deprecation_warning),
        (ResourceWarning, m.raise_resource_warning),
        (RuntimeWarning, m.raise_runtime_warning),
        (SyntaxWarning, m.raise_syntax_warning),
        (UnicodeWarning, m.raise_unicode_warning),
        (UserWarning, m.raise_user_warning),
    ],
)
def test_warning_categories(expected_category, module_function):
    with pytest.warns(Warning) as excinfo:
        module_function()

    assert issubclass(excinfo[0].category, expected_category)
    assert str(excinfo[0].message) == f"This is {expected_category.__name__}!"
