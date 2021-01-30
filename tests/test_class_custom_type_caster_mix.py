# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import class_custom_type_caster_mix as m


def test_make_unique_pointee():
    obj = m.NumberStore()
    assert obj.Get() == 5 # custom type_caster wins.
