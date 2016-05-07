#!/usr/bin/env python
from __future__ import print_function

from example import VectorInt, VectorA

v_int = VectorInt(2)
print( v_int.size() )

print( bool(v_int) )

v_int2 = VectorInt(2)
print( v_int == v_int2 )

v_int2[1] = 1
print( v_int != v_int2 )

v_int2.push_back(2)
v_int2.push_back(3)
print(v_int2)

v_a = VectorA()
