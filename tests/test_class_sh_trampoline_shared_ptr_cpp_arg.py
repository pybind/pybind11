import pytest

import pybind11_tests.class_sh_trampoline_shared_ptr_cpp_arg as m


def test_shared_ptr_cpp_arg():
    import weakref

    class PyChild(m.SpBase):
        def is_base_used(self):
            return False

    tester = m.SpBaseTester()

    obj = PyChild()
    objref = weakref.ref(obj)

    # Pass the last python reference to the C++ function
    tester.set_object(obj)
    del obj
    pytest.gc_collect()

    # python reference is still around since C++ has it now
    assert objref() is not None
    assert tester.is_base_used() is False
    assert tester.obj.is_base_used() is False
    assert tester.get_object() is objref()


def test_shared_ptr_cpp_prop():
    class PyChild(m.SpBase):
        def is_base_used(self):
            return False

    tester = m.SpBaseTester()

    # Set the last python reference as a property of the C++ object
    tester.obj = PyChild()
    pytest.gc_collect()

    # python reference is still around since C++ has it now
    assert tester.is_base_used() is False
    assert tester.has_python_instance() is True
    assert tester.obj.is_base_used() is False
    assert tester.obj.has_python_instance() is True


def test_shared_ptr_arg_identity():
    import weakref

    tester = m.SpBaseTester()

    obj = m.SpBase()
    objref = weakref.ref(obj)

    tester.set_object(obj)
    del obj
    pytest.gc_collect()

    # SMART_HOLDER_WIP: the behavior below is DIFFERENT from PR #2839
    # python reference is gone because it is not an Alias instance
    assert objref() is None
    assert tester.has_python_instance() is False


def test_shared_ptr_alias_nonpython():
    tester = m.SpBaseTester()

    # C++ creates the object, a python instance shouldn't exist
    tester.set_nonpython_instance()
    assert tester.is_base_used() is True
    assert tester.has_instance() is True
    assert tester.has_python_instance() is False

    # Now a python instance exists
    cobj = tester.get_object()
    assert cobj.has_python_instance()
    assert tester.has_instance() is True
    assert tester.has_python_instance() is True

    # Now it's gone
    del cobj
    pytest.gc_collect()
    assert tester.has_instance() is True
    assert tester.has_python_instance() is False

    # When we pass it as an arg to a new tester the python instance should
    # disappear because it wasn't created with an alias
    new_tester = m.SpBaseTester()

    cobj = tester.get_object()
    assert cobj.has_python_instance()

    new_tester.set_object(cobj)
    assert tester.has_python_instance() is True
    assert new_tester.has_python_instance() is True

    del cobj
    pytest.gc_collect()

    # Gone!
    assert tester.has_instance() is True
    assert tester.has_python_instance() is False
    assert new_tester.has_instance() is True
    assert new_tester.has_python_instance() is False


def test_shared_ptr_goaway():
    import weakref

    tester = m.SpGoAwayTester()

    obj = m.SpGoAway()
    objref = weakref.ref(obj)

    assert tester.obj is None

    tester.obj = obj
    del obj
    pytest.gc_collect()

    # python reference is no longer around
    assert objref() is None
    # C++ reference is still around
    assert tester.obj is not None


def test_infinite():
    tester = m.SpBaseTester()
    while True:
        tester.set_object(m.SpBase())
        break  # Comment out for manual leak checking (use `top` command).


@pytest.mark.parametrize(
    "pass_through_func", [m.pass_through_shd_ptr, m.pass_through_shd_ptr_release_gil]
)
def test_std_make_shared_factory(pass_through_func):
    class PyChild(m.SpBase):
        def __init__(self):
            super().__init__(0)

    obj = PyChild()
    while True:
        assert pass_through_func(obj) is obj
        break  # Comment out for manual leak checking (use `top` command).
