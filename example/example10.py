#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

import example
try:
    import numpy as np
except ImportError:
    print('NumPy missing')
    exit(0)

from example import vectorized_func
from example import vectorized_func2
from example import vectorized_func3

print(vectorized_func3(np.array(3+7j)))

for f in [vectorized_func, vectorized_func2]:
    print(f(1, 2, 3))
    print(f(np.array(1), np.array(2), 3))
    print(f(np.array([1, 3]), np.array([2, 4]), 3))
    print(f(np.array([[1, 3, 5], [7, 9, 11]]), np.array([[2, 4, 6], [8, 10, 12]]), 3))
    print(np.array([[1, 3, 5], [7, 9, 11]])* np.array([[2, 4, 6], [8, 10, 12]])*3)
    print(f(np.array([[1, 2, 3], [4, 5, 6]]), np.array([2, 3, 4]), 2))
    print(np.array([[1, 2, 3], [4, 5, 6]])* np.array([2, 3, 4])* 2)
    print(f(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[2], [3]]), 2))
    print(np.array([[1, 2, 3], [4, 5, 6]])* np.array([[2], [3]])* 2)

from example import selective_func

selective_func(np.array([1], dtype=np.int32))
selective_func(np.array([1.0], dtype=np.float32))
selective_func(np.array([1.0j], dtype=np.complex64))
