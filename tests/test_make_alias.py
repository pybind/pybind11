import pytest

from pybind11_tests import ConstructorStats

m = pytest.importorskip("pybind11_tests._make_alias")


def assert_name(mat):
    assert m.__name__ == "make_alias"
