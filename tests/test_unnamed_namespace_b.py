import pytest

from pybind11_tests import unnamed_namespace_b as m


@pytest.mark.xfail(
    "m.defined___clang__",
    reason="Known issue with all clang versions: https://github.com/pybind/pybind11/pull/4316",
    strict=False,
)
def test_have_class_any_struct():
    assert m.unnamed_namespace_b_any_struct is not None
