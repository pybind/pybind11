#!/usr/bin/env python
from __future__ import print_function

import numpy as np
from example import create_rec_simple, create_rec_packed


def check_eq(arr, data, dtype):
    np.testing.assert_equal(arr, np.array(data, dtype=dtype))

simple_dtype = np.dtype({'names': ['x', 'y', 'z'],
                         'formats': ['?', 'u4', 'f4'],
                         'offsets': [0, 4, 8]})
packed_dtype = np.dtype([('x', '?'), ('y', 'u4'), ('z', 'f4')])

for func, dtype in [(create_rec_simple, simple_dtype), (create_rec_packed, packed_dtype)]:
    arr = func(3)
    assert arr.dtype == dtype
    check_eq(arr, [(False, 0, 0.0), (True, 1, 1.5), (False, 2, 3.0)], simple_dtype)
    check_eq(arr, [(False, 0, 0.0), (True, 1, 1.5), (False, 2, 3.0)], packed_dtype)

    arr = func(0)
    assert arr.dtype == dtype
    check_eq(arr, [], simple_dtype)
    check_eq(arr, [], packed_dtype)
