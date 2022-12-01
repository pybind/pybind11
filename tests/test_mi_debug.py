import pytest

m = pytest.importorskip("pybind11_tests.mi_debug")


def test_vec():
    o = m.make_object()
    assert 5 == m.get_object_vec_size(o)
