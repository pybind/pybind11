from pybind11_tests import template_alias_base as m


def test_can_create_variable():
    v = m.S_std()
    print(v)


def test_can_return_variable():
    v = m.make_S()
    print(v)
