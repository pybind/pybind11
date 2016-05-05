#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import fixed_r, fixed_c
from example import fixed_passthrough_r, fixed_passthrough_c
from example import dense_r, dense_c
from example import dense_passthrough_r, dense_passthrough_c
from example import sparse_r, sparse_c
from example import sparse_passthrough_r, sparse_passthrough_c
import numpy as np

ref = np.array(
    [[0, 3, 0, 0, 0, 11],
     [22, 0, 0, 0, 17, 11],
     [7, 5, 0, 1, 0, 11],
     [0, 0, 0, 0, 0, 11],
     [0, 0, 14, 0, 8, 11]])


def check(mat):
    return 'OK' if np.sum(mat - ref) == 0 else 'NOT OK'

print("fixed_r = %s" % check(fixed_r()))
print("fixed_c = %s" % check(fixed_c()))
print("pt_r(fixed_r) = %s" % check(fixed_passthrough_r(fixed_r())))
print("pt_c(fixed_c) = %s" % check(fixed_passthrough_c(fixed_c())))
print("pt_r(fixed_c) = %s" % check(fixed_passthrough_r(fixed_c())))
print("pt_c(fixed_r) = %s" % check(fixed_passthrough_c(fixed_r())))

print("dense_r = %s" % check(dense_r()))
print("dense_c = %s" % check(dense_c()))
print("pt_r(dense_r) = %s" % check(dense_passthrough_r(dense_r())))
print("pt_c(dense_c) = %s" % check(dense_passthrough_c(dense_c())))
print("pt_r(dense_c) = %s" % check(dense_passthrough_r(dense_c())))
print("pt_c(dense_r) = %s" % check(dense_passthrough_c(dense_r())))

print("sparse_r = %s" % check(sparse_r()))
print("sparse_c = %s" % check(sparse_c()))
print("pt_r(sparse_r) = %s" % check(sparse_passthrough_r(sparse_r())))
print("pt_c(sparse_c) = %s" % check(sparse_passthrough_c(sparse_c())))
print("pt_r(sparse_c) = %s" % check(sparse_passthrough_r(sparse_c())))
print("pt_c(sparse_r) = %s" % check(sparse_passthrough_c(sparse_r())))
