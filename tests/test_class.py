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
