#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import Matrix

try:
    import numpy as np
except ImportError:
    print('NumPy missing')
    exit(0)

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
