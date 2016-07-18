#!/usr/bin/env python
from __future__ import print_function

from example import VectorInt, El, VectorEl, VectorVectorEl, VectorBool

v_int = VectorInt([0, 0])
print(len(v_int))

print(bool(v_int))

v_int2 = VectorInt([0, 0])
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
del v_int2[1:3]
print(v_int2)
del v_int2[0]
print(v_int2)

v_a = VectorEl()
v_a.append(El(1))
v_a.append(El(2))
print(v_a)

vv_a = VectorVectorEl()
vv_a.append(v_a)
vv_b = vv_a[0]
print(vv_b)

vv_c = VectorBool()
for i in range(10):
    vv_c.append(i % 2 == 0)
for i in range(10):
    if vv_c[i] != (i % 2 == 0):
        print("Error!")
print(vv_c)
