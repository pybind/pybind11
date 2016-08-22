try:
    import cPickle as pickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle

from pybind11_tests import Pickleable


def test_roundtrip():
    p = Pickleable("test_value")
    p.setExtra1(15)
    p.setExtra2(48)

    data = pickle.dumps(p, 2)  # Must use pickle protocol >= 2
    p2 = pickle.loads(data)
    assert p2.value() == p.value()
    assert p2.extra1() == p.extra1()
    assert p2.extra2() == p.extra2()
