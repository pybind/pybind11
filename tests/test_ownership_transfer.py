from pybind11_tests import ownership_transfer as m
from pybind11_tests import ConstructorStats

import pytest
import weakref


def define_child(name, base_type, stats_type):
    # Derived instance of `DefineBase<>` in C++.
    # `stats_type` is meant to enable us to use `ConstructorStats` exclusively for a Python class.

    class ChildT(base_type):
        def __init__(self, value):
            base_type.__init__(self, value)
            self.icstats = m.get_instance_cstats(ChildT.get_cstats(), self)
            self.icstats.track_created()

        def __del__(self):
            self.icstats.track_destroyed()

        def value(self):
            # Use a different value, so that we can detect slicing.
            return 10 * base_type.value(self)

        @staticmethod
        def get_cstats():
            return ConstructorStats.get(stats_type)

    ChildT.__name__ = name
    return ChildT


ChildBad = define_child('ChildBad', m.BaseBad, m.ChildBadStats)
Child = define_child('Child', m.Base, m.ChildStats)

ChildBadUnique = define_child(
    'ChildBadUnique', m.BaseBadUnique, m.ChildBadUniqueStats)
ChildUnique = define_child(
    'ChildUnique', m.BaseUnique, m.ChildUniqueStats)


# TODO(eric.cousineau): See if this is at all possibly on PyPy.
# Placing `pytest.gc_collect` near `del` statements indicates that we are not
# capturing deletion properly.
@pytest.unsupported_on_pypy
def test_shared_ptr_derived_slicing(capture):
    from sys import getrefcount

    # [ Bad ]
    cstats = ChildBad.get_cstats()
    # Create instance in move container to permit releasing.
    obj = ChildBad(10)
    obj_weak = weakref.ref(obj)
    # This will release the reference, the refcount will drop to zero, and Python will destroy it.
    c = m.BaseBadContainer(obj)
    del obj
    # We will have lost the derived Python instance.
    assert obj_weak() is None
    # Check stats:
    assert cstats.alive() == 0
    # As an additional check, we will try to query the value from the container's value.
    # This should have been 100 if the trampoline had retained its Python portion.
    assert c.get().value() == 10
    # Destroy references.
    del c

    # [ Good ]
    # See above for setup.
    cstats = Child.get_cstats()
    # Try a temporary setup.
    c = m.BaseContainer(Child(10))
    assert cstats.alive() == 1
    obj = c.release()
    del c
    assert cstats.alive() == 1
    assert obj.value() == 100
    del obj
    assert cstats.alive() == 0
    # Use something more permanent.
    obj = Child(10)
    obj_weak = weakref.ref(obj)
    assert cstats.alive() == 1
    c = m.BaseContainer(obj)
    del obj
    assert cstats.alive() == 1
    # We now still have a reference to the object. py::wrapper<> will intercept Python's
    # attempt to destroy `obj`, is aware the `shared_ptr<>.use_count() > 1`, and will increase
    # the ref count by transferring a new reference to `py::wrapper<>` (thus reviving the object,
    # per Python's documentation of __del__).
    assert obj_weak() is not None
    assert cstats.alive() == 1
    # This goes from C++ -> Python, and then Python -> C++ once this statement has finished.
    assert c.get().value() == 100
    assert cstats.alive() == 1
    # Destroy references (effectively in C++), and ensure that we have the desired behavior.
    del c
    assert cstats.alive() == 0

    # Ensure that we can pass it from Python -> C++ -> Python, and ensure that C++ does not think
    # that it has ownership.
    obj = Child(10)
    c = m.BaseContainer(obj)
    del obj
    assert cstats.alive() == 1
    obj = c.get()
    # Now that we have it in Python, there should only be 1 Python reference, since
    # py::wrapper<> in C++ should have released its reference.
    assert getrefcount(obj) == 2
    del c
    assert cstats.alive() == 1
    assert obj.value() == 100
    del obj
    assert cstats.alive() == 0


@pytest.unsupported_on_pypy
def test_unique_ptr_derived_slicing(capture):
    # [ Bad ]
    cstats = ChildBadUnique.get_cstats()
    # Create instance in move container to permit releasing.
    try:
        c = m.BaseBadUniqueContainer(ChildBadUnique(10))
    except RuntimeError as e:
        assert "wrapper<>" in str(e)
    # We lost this portion to slicing.
    assert cstats.alive() == 0

    # [ Good ]
    cstats = ChildUnique.get_cstats()
    # Try a temporary setup.
    c = m.BaseUniqueContainer(ChildUnique(10))
    assert cstats.alive() == 1
    obj = c.release()
    del c
    assert cstats.alive() == 1
    assert obj.value() == 100
    del obj
    assert cstats.alive() == 0

    # Ensure that we can pass between Python -> C++ -> Python.
    obj = m.BaseUniqueContainer(ChildUnique(10)).release()
    obj = m.BaseUniqueContainer(obj).release()
    assert obj.value() == 100
    del obj
    assert cstats.alive() == 0
