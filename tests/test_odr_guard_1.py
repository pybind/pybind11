import pybind11_tests.odr_guard_1 as m


def test_type_mrc_to_python():
    assert m.type_mrc_to_python() == 1111


def test_type_mrc_from_python():
    assert m.type_mrc_from_python("ignored") == 111
