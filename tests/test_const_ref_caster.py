# -*- coding: utf-8 -*-
import pytest

import env  # noqa: F401

from pybind11_tests import const_ref_caster as m


def test_takes():
    x = False
    assert m.takes(x)

    assert m.takes_ptr(x)
    assert m.takes_ref(x)
    assert m.takes_ref_wrap(x)

    assert m.takes_const_ptr(x)
    assert m.takes_const_ref(x)
    assert m.takes_const_ref_wrap(x)
