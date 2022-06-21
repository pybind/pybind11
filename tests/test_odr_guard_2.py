import pybind11_tests.odr_guard_2 as m


def test_type_mrc_to_python():
    assert m.type_mrc_to_python() in (202 + 2020, 202 + 1010)


def test_type_mrc_from_python():
    assert m.type_mrc_from_python("ignored") in (200 + 22, 200 + 11)
