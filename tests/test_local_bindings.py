import pytest

from pybind11_tests import local_bindings as m


def test_local_bindings():
    """Tests that duplicate `py::module_local` class bindings work across modules"""

    # Make sure we can load the second module with the conflicting (but local) definition:
    import pybind11_cross_module_tests as cm

    i1 = m.LocalType(5)
    assert i1.get() == 4
    assert i1.get3() == 8

    i2 = cm.LocalType(10)
    assert i2.get() == 11
    assert i2.get2() == 12

    assert not hasattr(i1, 'get2')
    assert not hasattr(i2, 'get3')

    assert m.local_value(i1) == 5
    assert cm.local_value(i2) == 10

    with pytest.raises(TypeError) as excinfo:
        m.local_value(i2)
    assert "incompatible function arguments" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        cm.local_value(i1)
    assert "incompatible function arguments" in str(excinfo.value)


def test_nonlocal_failure():
    """Tests that attempting to register a non-local type in multiple modules fails"""
    import pybind11_cross_module_tests as cm

    with pytest.raises(RuntimeError) as excinfo:
        cm.register_nonlocal()
    assert str(excinfo.value) == 'generic_type: type "NonLocalType" is already registered!'


def test_duplicate_local():
    """Tests expected failure when registering a class twice with py::local in the same module"""
    with pytest.raises(RuntimeError) as excinfo:
        m.register_local_external()
    import pybind11_tests
    assert str(excinfo.value) == (
        'generic_type: type "LocalExternal" is already registered!'
        if hasattr(pybind11_tests, 'class_') else 'test_class not enabled')


def test_stl_bind_local():
    import pybind11_cross_module_tests as cm

    v1, v2 = m.LocalVec(), cm.LocalVec()
    v1.append(m.LocalType(1))
    v1.append(m.LocalType(2))
    v2.append(cm.LocalType(1))
    v2.append(cm.LocalType(2))

    with pytest.raises(TypeError):
        v1.append(cm.LocalType(3))
    with pytest.raises(TypeError):
        v2.append(m.LocalType(3))

    assert [i.get() for i in v1] == [0, 1]
    assert [i.get() for i in v2] == [2, 3]

    v3, v4 = m.NonLocalVec(), cm.NonLocalVec2()
    v3.append(m.NonLocalType(1))
    v3.append(m.NonLocalType(2))
    v4.append(m.NonLocal2(3))
    v4.append(m.NonLocal2(4))

    assert [i.get() for i in v3] == [1, 2]
    assert [i.get() for i in v4] == [13, 14]

    d1, d2 = m.LocalMap(), cm.LocalMap()
    d1["a"] = v1[0]
    d1["b"] = v1[1]
    d2["c"] = v2[0]
    d2["d"] = v2[1]
    assert {i: d1[i].get() for i in d1} == {'a': 0, 'b': 1}
    assert {i: d2[i].get() for i in d2} == {'c': 2, 'd': 3}


def test_stl_bind_global():
    import pybind11_cross_module_tests as cm

    with pytest.raises(RuntimeError) as excinfo:
        cm.register_nonlocal_map()
    assert str(excinfo.value) == 'generic_type: type "NonLocalMap" is already registered!'

    with pytest.raises(RuntimeError) as excinfo:
        cm.register_nonlocal_vec()
    assert str(excinfo.value) == 'generic_type: type "NonLocalVec" is already registered!'

    with pytest.raises(RuntimeError) as excinfo:
        cm.register_nonlocal_map2()
    assert str(excinfo.value) == 'generic_type: type "NonLocalMap2" is already registered!'


def test_mixed_local_global():
    """Local types take precedence over globally registered types: a module with a `module_local`
    type can be registered even if the type is already registered globally.  With the module,
    casting will go to the local type; outside the module casting goes to the global type."""
    import pybind11_cross_module_tests as cm
    m.register_mixed_global()
    m.register_mixed_local()

    a = []
    a.append(m.MixedGlobalLocal(1))
    a.append(m.MixedLocalGlobal(2))
    a.append(m.get_mixed_gl(3))
    a.append(m.get_mixed_lg(4))

    assert [x.get() for x in a] == [101, 1002, 103, 1004]

    cm.register_mixed_global_local()
    cm.register_mixed_local_global()
    a.append(m.MixedGlobalLocal(5))
    a.append(m.MixedLocalGlobal(6))
    a.append(cm.MixedGlobalLocal(7))
    a.append(cm.MixedLocalGlobal(8))
    a.append(m.get_mixed_gl(9))
    a.append(m.get_mixed_lg(10))
    a.append(cm.get_mixed_gl(11))
    a.append(cm.get_mixed_lg(12))

    assert [x.get() for x in a] == \
        [101, 1002, 103, 1004, 105, 1006, 207, 2008, 109, 1010, 211, 2012]


def test_internal_locals_differ():
    """Makes sure the internal local type map differs across the two modules"""
    import pybind11_cross_module_tests as cm
    assert m.local_cpp_types_addr() != cm.local_cpp_types_addr()


def test_stl_caster_vs_stl_bind(msg):
    """One module uses a generic vector caster from `<pybind11/stl.h>` while the other
    exports `std::vector<int>` via `py:bind_vector` and `py::module_local`"""
    import pybind11_cross_module_tests as cm

    v1 = cm.VectorInt([1, 2, 3])
    assert m.load_vector_via_caster(v1) == 6
    assert cm.load_vector_via_binding(v1) == 6

    v2 = [1, 2, 3]
    assert m.load_vector_via_caster(v2) == 6
    with pytest.raises(TypeError) as excinfo:
        cm.load_vector_via_binding(v2) == 6
    assert msg(excinfo.value) == """
    load_vector_via_binding(): incompatible function arguments. The following argument types are supported:
        1. (arg0: pybind11_cross_module_tests.VectorInt) -> int

    Invoked with: [1, 2, 3]
    """  # noqa: E501 line too long
