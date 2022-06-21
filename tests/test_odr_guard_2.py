import pybind11_tests.odr_guard_2 as m


def test_type_mrc_to_python():
    assert m.type_mrc_to_python() == 2222


def test_type_mrc_from_python():
    assert m.type_mrc_from_python("ignored") == 222
