from pybind11_tests import mi_debug as m


def test_get_vec_size_raw_ptr_base0():
    obj = m.make_derived_as_base0()
    assert m.get_vec_size_raw_ptr_base0(obj) == 5


def test_get_vec_size_raw_ptr_derived():
    obj = m.make_derived_as_base0()
    assert m.get_vec_size_raw_ptr_derived(obj) == 5


def test_get_vec_size_shared_ptr_derived():
    obj = m.make_derived_as_base0()
    assert m.get_vec_size_shared_ptr_derived(obj) == 5
