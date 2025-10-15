from __future__ import annotations

import gc
import sys
import sysconfig
import types
import weakref

import pytest

import env
from pybind11_tests import class_cross_module_use_after_one_module_dealloc as m

is_python_3_13_free_threaded = (
    env.CPYTHON
    and sysconfig.get_config_var("Py_GIL_DISABLED")
    and (3, 13) <= sys.version_info < (3, 14)
)


def delattr_and_ensure_destroyed(*specs):
    wrs = []
    for mod, name in specs:
        wrs.append(weakref.ref(getattr(mod, name)))
        delattr(mod, name)

    for _ in range(5):
        gc.collect()
        if all(wr() is None for wr in wrs):
            break
    else:
        pytest.fail(
            f"Could not delete bindings such as {next(wr for wr in wrs if wr() is not None)!r}"
        )


@pytest.mark.skipif("env.PYPY or env.GRAALPY or is_python_3_13_free_threaded")
def test_cross_module_use_after_one_module_dealloc():
    # This is a regression test for a bug that occurred during development of
    # internals::registered_types_cpp_fast (see #5842). registered_types_cpp_fast maps
    # &typeid(T) to a raw non-owning pointer to a Python type object. If two DSOs both
    # look up the same global type, they will create two separate entries in
    # registered_types_cpp_fast, which will look like:
    # +=========================================+
    # |&typeid(T) from DSO 1|type object pointer|
    # |&typeid(T) from DSO 2|type object pointer|
    # +=========================================+
    #
    # Then, if the type object is destroyed and we don't take extra steps to clean up
    # the table thoroughly, the first row of the table will be cleaned up but the second
    # one will contain a dangling pointer to the old type object. Further lookups from
    # DSO 2 will then return that dangling pointer, which will cause use-after-frees.

    import pybind11_cross_module_tests as cm

    module_scope = types.ModuleType("module_scope")
    instance = m.register_and_instantiate_cross_dso_class(module_scope)
    cm.consume_cross_dso_class(instance)

    del instance
    delattr_and_ensure_destroyed((module_scope, "CrossDSOClass"))

    # Make sure that CrossDSOClass gets allocated at a different address.
    m.register_unrelated_class(module_scope)

    instance = m.register_and_instantiate_cross_dso_class(module_scope)
    cm.consume_cross_dso_class(instance)
