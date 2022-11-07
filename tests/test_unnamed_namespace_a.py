# NOTE: This test relies on pytest SORT ORDER:
#       test_unnamed_namespace_a.py imported before test_unnamed_namespace_b.py

from pybind11_tests import unnamed_namespace_a as m


def test_have_class_any_struct():
    assert m.unnamed_namespace_a_any_struct is not None
