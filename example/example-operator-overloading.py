#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import Vector2, Vector

v1 = Vector2(1, 2)
v2 = Vector(3, -1)

print("v1    = " + str(v1))
print("v2    = " + str(v2))
print("v1+v2 = " + str(v1+v2))
print("v1-v2 = " + str(v1-v2))
print("v1-8  = " + str(v1-8))
print("v1+8  = " + str(v1+8))
print("v1*8  = " + str(v1*8))
print("v1/8  = " + str(v1/8))
print("8-v1  = " + str(8-v1))
print("8+v1  = " + str(8+v1))
print("8*v1  = " + str(8*v1))
print("8/v1  = " + str(8/v1))

v1 += v2
v1 *= 2

print("(v1+v2)*2 = " + str(v1))

from example import ConstructorStats
cstats = ConstructorStats.get(Vector2)
print("Instances not destroyed:", cstats.alive())
v1 = None
print("Instances not destroyed:", cstats.alive())
v2 = None
print("Instances not destroyed:", cstats.alive())
print("Constructor values:", cstats.values())
print("Default constructions:", cstats.default_constructions)
print("Copy constructions:", cstats.copy_constructions)
print("Move constructions:", cstats.move_constructions >= 10)
print("Copy assignments:", cstats.copy_assignments)
print("Move assignments:", cstats.move_assignments)
