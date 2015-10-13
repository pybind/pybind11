#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

import example

print(example.__name__)
print(example.submodule.__name__)

from example.submodule import *
from example import OD

submodule_func()

b = B()
print(b.get_a1())
print(b.a1)
print(b.get_a2())
print(b.a2)

b.a1 = A(42)
b.a2 = A(43)

print(b.get_a1())
print(b.a1)
print(b.get_a2())
print(b.a2)

print(OD([(1, 'a'), (2, 'b')]))
