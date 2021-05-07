# -*- coding: utf-8 -*-
import pytest

import env  # noqa: F401

from pybind11_tests import pickling_trampoline as m

try:
    import cPickle as pickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle


class SimplePyDerived(m.SimpleBase):
    pass


def test_roundtrip_simple_py_derived():
    p = SimplePyDerived()
    p.num = 202
    p.stored_in_dict = 303
    data = pickle.dumps(p, pickle.HIGHEST_PROTOCOL)
    p2 = pickle.loads(data)
    assert p2.num == 202
    assert p2.stored_in_dict == 303


def test_roundtrip_simple_cpp_derived():
    p = m.make_SimpleCppDerivedAsBase()
    p.num = 404
    if not env.PYPY:
        with pytest.raises(AttributeError):
            # To ensure that future changes do not accidentally invalidate this unit test.
            p.__dict__
    data = pickle.dumps(p, pickle.HIGHEST_PROTOCOL)
    p2 = pickle.loads(data)
    assert p2.num == 404
