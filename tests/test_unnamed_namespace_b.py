from pybind11_tests import unnamed_namespace_b as m


def test_inspect():
    assert m.name == "UB"
