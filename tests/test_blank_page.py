from pybind11_tests import blank_page as m


def test_property():
    options = m.Options()
    options.simple_value = 100
