#!/usr/bin/env python
from __future__ import print_function

import numpy as np
from example import create_rec_simple


def check_eq(arr, data, dtype):
    np.testing.assert_equal(arr, np.array(data, dtype=dtype))

dtype = np.dtype({'names': ['x', 'y', 'z'],
                  'formats': ['?', 'u4', 'f4'],
                  'offsets': [0, 4, 8]})
base_dtype = np.dtype([('x', '?'), ('y', 'u4'), ('z', 'f4')])

arr = create_rec_simple(3)
assert arr.dtype == dtype
check_eq(arr, [(False, 0, 0.0), (True, 1, 1.5), (False, 2, 3.0)], dtype)
check_eq(arr, [(False, 0, 0.0), (True, 1, 1.5), (False, 2, 3.0)], base_dtype)
