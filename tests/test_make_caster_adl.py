# -*- coding: utf-8 -*-

from pybind11_tests import make_caster_adl as m


def test_basic():
    assert m.num_one() == 101
    assert m.num_two() == 202
