# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import unique_ptr_member as m


def test_cpp_pattern():
    res = m.cpp_pattern()
    assert res == 10


def test_pointee_and_ptr_owner():
    obj = m.pointee()
    assert obj.get_int() == 213
    m.ptr_owner(obj)
    with pytest.raises(ValueError) as exc_info:
        obj.get_int()
    assert str(exc_info.value).startswith("Missing value for wrapped C++ type ")
