# -*- coding: utf-8 -*-
from pybind11_tests import class_sh_property as m


# Reduced from:
# https://github.com/google/clif/blob/c371a6d4b28d25d53a16e6d2a6d97305fb1be25a/clif/testing/python/nested_fields_test.py#L56
def test_inner_access_after_disowning_outer():
    outer = m.Outer()
    inner = outer.field
    assert inner.value == -99
    m.DisownOuter(outer)
    # assert inner.value == -99 # AddressSanitizer: heap-use-after-free
