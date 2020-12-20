# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import smart_ptr_private_first_base as m

def test_make_pass():
    d = m.make_shared_drvd()
    i = m.pass_shared_base(d)
    assert i == 200
