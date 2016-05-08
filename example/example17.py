#!/usr/bin/env python
from __future__ import print_function

from example import VectorInt, VectorA, A

v_int = VectorInt(2)
print(len(v_int))

print(bool(v_int))

v_int2 = VectorInt(2)
print(v_int == v_int2)

v_int2[1] = 1
print(v_int != v_int2)

v_int2.append(2)
v_int2.append(3)
v_int2.insert(0, 1)
v_int2.insert(0, 2)
v_int2.insert(0, 3)
print(v_int2)

v_int.append(99)
v_int2[2:-2] = v_int
print(v_int2)

v_a = VectorA()
v_a.append(A(1))
v_a.append(A(2))
print(v_a)
