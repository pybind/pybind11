#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import Matrix

try:
    import numpy as np
except ImportError:
    # NumPy missing: skip test
    exit(99)

m = Matrix(5, 5)

print(m[2, 3])
m[2, 3] = 4
print(m[2, 3])

m2 = np.array(m, copy=False)
print(m2)
print(m2[2, 3])
m2[2, 3] = 5
print(m[2, 3])

m3 = np.array([[1,2,3],[4,5,6]]).astype(np.float32)
print(m3)
m4 = Matrix(m3)
for i in range(m4.rows()):
    for j in range(m4.cols()):
        print(m4[i, j], end = ' ')
    print()

from example import ConstructorStats
cstats = ConstructorStats.get(Matrix)
print("Instances not destroyed:", cstats.alive())
m = m4 = None
print("Instances not destroyed:", cstats.alive())
m2 = None # m2 holds an m reference
print("Instances not destroyed:", cstats.alive())
print("Constructor values:", cstats.values())
print("Copy constructions:", cstats.copy_constructions)
#print("Move constructions:", cstats.move_constructions >= 0) # Don't invoke any
print("Copy assignments:", cstats.copy_assignments)
print("Move assignments:", cstats.move_assignments)
