# -*- coding: utf-8 -*-
import pytest

import pybind11_tests.trampoline_shared_ptr_cpp_arg as m


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
    assert tester.obj.is_base_used() is False


def test_shared_ptr_arg_identity():
    import weakref

    tester = m.SpBaseTester()

    obj = m.SpBase()
    objref = weakref.ref(obj)

    tester.set_object(obj)
    del obj
    pytest.gc_collect()

    # python reference is still around since C++ has it
    assert objref() is not None
    assert tester.get_object() is objref()

    # python reference disappears once the C++ object releases it
    tester.set_object(None)
    pytest.gc_collect()
    assert objref() is None
