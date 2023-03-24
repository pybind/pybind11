from pybind11_tests import property_stl as m


def test_cpy_after_ref():
    h = m.VectorFieldHolder()
    c1 = h.vec_fld_hld_cpy
    c2 = h.vec_fld_hld_cpy
    assert repr(c2) != repr(c1)
    r1 = h.vec_fld_hld_ref
    assert repr(r1) != repr(c2)
    assert repr(r1) != repr(c1)
    r2 = h.vec_fld_hld_ref
    assert repr(r2) == repr(r1)
    c3 = h.vec_fld_hld_cpy
    assert repr(c3) == repr(r1)  # SURPRISE!


def test_persistent_holder():
    h = m.VectorFieldHolder()
    c0 = h.vec_fld_hld_cpy[0]  # Must be first. See test_cpy_after_ref().
    r0 = h.vec_fld_hld_ref[0]  # Must be second.
    assert c0.fld.wrapped_int == 300
    assert r0.fld.wrapped_int == 300
    h.reset_at(0, 400)
    assert c0.fld.wrapped_int == 300
    assert r0.fld.wrapped_int == 400


def test_temporary_holder_keep_alive():
    r0 = m.VectorFieldHolder().vec_fld_hld_ref[0]
    assert r0.fld.wrapped_int == 300
