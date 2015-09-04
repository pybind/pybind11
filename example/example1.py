#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import Example1

instance1 = Example1()
instance2 = Example1(32)
instance1.add1(instance2)
instance1.add2(instance2)
instance1.add3(instance2)
instance1.add4(instance2)
instance1.add5(instance2)
instance1.add6(32)
instance1.add7(32)
instance1.add8(32)
instance1.add9(32)
instance1.add10(32)

print("Instance 1: " + str(instance1))
print("Instance 2: " + str(instance2))

print(instance1.self1())
print(instance1.self2())
print(instance1.self3())
print(instance1.self4())
print(instance1.self5())
print(instance1.internal1())
print(instance1.internal2())
print(instance1.internal3())
print(instance1.internal4())
print(instance1.internal5())

print("Instance 1, direct access = %i" % instance1.value)
instance1.value = 100
print("Instance 1: " + str(instance1))
