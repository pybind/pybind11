#!/usr/bin/env python3
import sys
sys.path.append('.')

import example
import numpy as np

from example import vectorized_func
from example import vectorized_func2

for f in [vectorized_func, vectorized_func2]:
    print(f(1, 2, 3))
    print(f(np.array(1), np.array(2), 3))
    print(f(np.array([1, 3]), np.array([2, 4]), 3))
    print(f(np.array([[1, 3, 5], [7, 9, 11]]), np.array([[2, 4, 6], [8, 10, 12]]), 3))
    print(np.array([[1, 3, 5], [7, 9, 11]])* np.array([[2, 4, 6], [8, 10, 12]])*3)
