from pybind11_tests import blank_page as m


def test_property():
    options = m.Options()
    setter_return = options.simple_value = 100
    assert isinstance(setter_return, int)
    assert setter_return == 100
    sv = options.simple_value
    assert sv == 100
