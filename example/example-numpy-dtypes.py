#!/usr/bin/env python
from __future__ import print_function

import unittest
import numpy as np
from example import (
    create_rec_simple, create_rec_packed, create_rec_nested, print_format_descriptors,
    print_rec_simple, print_rec_packed, print_rec_nested, print_dtypes, get_format_unbound,
    create_rec_partial, create_rec_partial_nested, create_string_array, print_string_array
)


def check_eq(arr, data, dtype):
    np.testing.assert_equal(arr, np.array(data, dtype=dtype))

unittest.TestCase().assertRaisesRegex(
    RuntimeError, 'unsupported buffer format', get_format_unbound)

print_format_descriptors()
print_dtypes()

simple_dtype = np.dtype({'names': ['x', 'y', 'z'],
                         'formats': ['?', 'u4', 'f4'],
                         'offsets': [0, 4, 8]})
packed_dtype = np.dtype([('x', '?'), ('y', 'u4'), ('z', 'f4')])

elements = [(False, 0, 0.0), (True, 1, 1.5), (False, 2, 3.0)]

for func, dtype in [(create_rec_simple, simple_dtype), (create_rec_packed, packed_dtype)]:
    arr = func(0)
    assert arr.dtype == dtype
    check_eq(arr, [], simple_dtype)
    check_eq(arr, [], packed_dtype)

    arr = func(3)
    assert arr.dtype == dtype
    check_eq(arr, elements, simple_dtype)
    check_eq(arr, elements, packed_dtype)

    if dtype == simple_dtype:
        print_rec_simple(arr)
    else:
        print_rec_packed(arr)


arr = create_rec_partial(3)
print(arr.dtype)
partial_dtype = arr.dtype
assert '' not in arr.dtype.fields
assert partial_dtype.itemsize > simple_dtype.itemsize
check_eq(arr, elements, simple_dtype)
check_eq(arr, elements, packed_dtype)

arr = create_rec_partial_nested(3)
print(arr.dtype)
assert '' not in arr.dtype.fields
assert '' not in arr.dtype.fields['a'][0].fields
assert arr.dtype.itemsize > partial_dtype.itemsize
np.testing.assert_equal(arr['a'], create_rec_partial(3))

nested_dtype = np.dtype([('a', simple_dtype), ('b', packed_dtype)])

arr = create_rec_nested(0)
assert arr.dtype == nested_dtype
check_eq(arr, [], nested_dtype)

arr = create_rec_nested(3)
assert arr.dtype == nested_dtype
check_eq(arr, [((False, 0, 0.0), (True, 1, 1.5)),
               ((True, 1, 1.5), (False, 2, 3.0)),
               ((False, 2, 3.0), (True, 3, 4.5))], nested_dtype)
print_rec_nested(arr)

assert create_rec_nested.__doc__.strip().endswith('numpy.ndarray[dtype=NestedStruct]')

arr = create_string_array(True)
print(arr.dtype)
print_string_array(arr)
dtype = arr.dtype
assert arr['a'].tolist() == [b'', b'a', b'ab', b'abc']
assert arr['b'].tolist() == [b'', b'a', b'ab', b'abc']
arr = create_string_array(False)
assert dtype == arr.dtype
