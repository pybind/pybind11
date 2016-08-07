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

from example import ConstructorStats

cstats = [ConstructorStats.get(A), ConstructorStats.get(B)]
print("Instances not destroyed:", [x.alive() for x in cstats])
b = None
print("Instances not destroyed:", [x.alive() for x in cstats])
print("Constructor values:", [x.values() for x in cstats])
print("Default constructions:", [x.default_constructions for x in cstats])
print("Copy constructions:", [x.copy_constructions for x in cstats])
#print("Move constructions:", [x.move_constructions >= 0 for x in cstats]) # Don't invoke any
print("Copy assignments:", [x.copy_assignments for x in cstats])
print("Move assignments:", [x.move_assignments for x in cstats])
