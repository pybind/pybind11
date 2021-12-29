# -*- coding: utf-8 -*-
from pybind11_tests import class_sh_property as m


def test_inner_access_after_disowning_outer():
    outer = m.Outer()
    inner = outer.field
    assert inner.value == -99
    m.DisownOuter(outer)
    # assert inner.value == -99 # AddressSanitizer: heap-use-after-free
