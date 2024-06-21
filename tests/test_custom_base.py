from pybind11_tests import custom_base as m


def test_cb_base():
    b = m.create_base()

    assert isinstance(b, m.Base)
    assert b.i == 5

    assert m.base_i(b) == 5


def test_cb_derived():
    d = m.create_derived()

    assert isinstance(d, m.Derived)
    assert isinstance(d, m.Base)

    assert d.i == 5
    assert d.j == 6

    assert m.base_i(d) == 5
    assert m.derived_j(d) == 6
