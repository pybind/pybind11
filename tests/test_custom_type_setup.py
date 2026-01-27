from __future__ import annotations

import gc
import os
import sys
import weakref

import pytest

import env
import pybind11_tests
from pybind11_tests import custom_type_setup as m


@pytest.fixture
def gc_tester():
    """Tests that an object is garbage collected.

    Assumes that any unreferenced objects are fully collected after calling
    `gc.collect()`.  That is true on CPython, but does not appear to reliably
    hold on PyPy.
    """

    weak_refs = []

    def add_ref(obj):
        # PyPy does not support `gc.is_tracked`.
        if hasattr(gc, "is_tracked"):
            assert gc.is_tracked(obj)
        weak_refs.append(weakref.ref(obj))

    yield add_ref

    gc.collect()
    for ref in weak_refs:
        assert ref() is None


# PyPy does not seem to reliably garbage collect.
@pytest.mark.skipif("env.PYPY or env.GRAALPY")
def test_self_cycle(gc_tester):
    obj = m.ContainerOwnsPythonObjects()
    obj.append(obj)
    gc_tester(obj)


# PyPy does not seem to reliably garbage collect.
@pytest.mark.skipif("env.PYPY or env.GRAALPY")
def test_indirect_cycle(gc_tester):
    obj = m.ContainerOwnsPythonObjects()
    obj.append([obj])
    gc_tester(obj)


@pytest.mark.skipif(
    env.IOS or sys.platform.startswith("emscripten"),
    reason="Requires subprocess support",
)
@pytest.mark.skipif("env.PYPY or env.GRAALPY")
def test_py_cast_useable_on_shutdown():
    """Test that py::cast works during interpreter shutdown.

    See PR #5972 and https://github.com/pybind/pybind11/pull/5958#discussion_r2717645230.
    """
    env.check_script_success_in_subprocess(
        f"""
        import sys

        sys.path.insert(0, {os.path.dirname(env.__file__)!r})
        sys.path.insert(0, {os.path.dirname(pybind11_tests.__file__)!r})

        from pybind11_tests import custom_type_setup as m

        # Create a self-referential cycle that will be collected during shutdown.
        # The tp_traverse and tp_clear callbacks call py::cast, which requires
        # internals to still be valid.
        obj = m.ContainerOwnsPythonObjects()
        obj.append(obj)

        # Add weakref callbacks that verify the capsule is still alive when the
        # pybind11 object is garbage collected during shutdown.
        m.add_gc_checkers_with_weakrefs(obj)
        """
    )
