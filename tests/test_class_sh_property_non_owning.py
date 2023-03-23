from pybind11_tests import class_sh_property_non_owning as m


def test_persistent_holder():
    h = m.DataFieldsHolder(2)
    c = h.vec_at(0).core_fld
    assert c.int_value == 13
    assert h.vec_at(1).core_fld.int_value == 24


def test_temporary_holder():
    d = m.DataFieldsHolder(2).vec_at(1)
    c = d.core_fld
    assert c.int_value == 24
