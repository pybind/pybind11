from __future__ import print_function
import sys

sys.path.append('.')

from example import Pickleable

try:
    import cPickle as pickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle

p = Pickleable("test_value")
p.setExtra1(15)
p.setExtra2(48)

data = pickle.dumps(p, 2)  # Must use pickle protocol >= 2
print("%s %i %i" % (p.value(), p.extra1(), p.extra2()))

p2 = pickle.loads(data)
print("%s %i %i" % (p2.value(), p2.extra1(), p2.extra2()))
