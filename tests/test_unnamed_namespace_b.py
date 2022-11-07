# NOTE: This test relies on pytest SORT ORDER:
#       test_unnamed_namespace_a.py imported before test_unnamed_namespace_b.py

import pytest

from pybind11_tests import unnamed_namespace_b as m


@pytest.mark.xfail(
    "m.defined___clang__",
    reason="Known issue with all clang versions: https://github.com/pybind/pybind11/pull/4316",
    strict=True,
)
def test_have_class_any_struct():
    assert m.unnamed_namespace_b_any_struct is not None
