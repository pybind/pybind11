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
from example import double_row, double_col
from example import double_mat_cm, double_mat_rm
try:
    import numpy as np
    import scipy
except ImportError:
    # NumPy missing: skip test
    exit(99)

ref = np.array(
    [[0, 3, 0, 0, 0, 11],
     [22, 0, 0, 0, 17, 11],
     [7, 5, 0, 1, 0, 11],
     [0, 0, 0, 0, 0, 11],
     [0, 0, 14, 0, 8, 11]])


def check(mat):
    return 'OK' if np.sum(abs(mat - ref)) == 0 else 'NOT OK'

print("should_give_NOT_OK = %s" % check(ref[:, ::-1]))

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

def check_got_vs_ref(got_x, ref_x):
    return 'OK' if np.array_equal(got_x, ref_x) else 'NOT OK'

counting_mat = np.arange(9.0, dtype=np.float32).reshape((3, 3))
first_row = counting_mat[0, :]
first_col = counting_mat[:, 0]

print("double_row(first_row) = %s" % check_got_vs_ref(double_row(first_row), 2.0 * first_row))
print("double_col(first_row) = %s" % check_got_vs_ref(double_col(first_row), 2.0 * first_row))
print("double_row(first_col) = %s" % check_got_vs_ref(double_row(first_col), 2.0 * first_col))
print("double_col(first_col) = %s" % check_got_vs_ref(double_col(first_col), 2.0 * first_col))

counting_3d = np.arange(27.0, dtype=np.float32).reshape((3, 3, 3))
slices = [counting_3d[0, :, :], counting_3d[:, 0, :], counting_3d[:, :, 0]]

for slice_idx, ref_mat in enumerate(slices):
    print("double_mat_cm(%d) = %s" % (slice_idx, check_got_vs_ref(double_mat_cm(ref_mat), 2.0 * ref_mat)))
    print("double_mat_rm(%d) = %s" % (slice_idx, check_got_vs_ref(double_mat_rm(ref_mat), 2.0 * ref_mat)))
