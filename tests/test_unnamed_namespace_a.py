from pybind11_tests import unnamed_namespace_a as m


def test_have_type():
    assert hasattr(m, "unnamed_namespace_a_any_struct")
