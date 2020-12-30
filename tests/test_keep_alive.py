# -*- coding: utf-8 -*-
# N.B. This is only focused on CPython, so using gc directly, rather than
# `pytest.gc_collect()`.
import gc
import weakref

from pybind11_tests import keep_alive as m


def test_keep_alive_cycle():
    # See #2761.
    o1 = m.SimpleClass()
    wr1 = weakref.ref(o1)
    o2 = m.SimpleClass()
    wr2 = weakref.ref(o2)
    assert wr1() is not None
    assert wr2() is not None

    # Add a direct cycle.
    m.keep_alive_impl(o1, o2)
    m.keep_alive_impl(o2, o1)

    del o1
    del o2
    gc.collect()

    # This shows that py::keep_alive will leak objects :(
    assert wr1() is not None
    assert wr2() is not None
