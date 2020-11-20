# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import unique_ptr_member as m


def test_make_unique_pointee():
    m.to_cout("")
    obj = m.make_unique_pointee()
    assert obj.get_int() == 213
    m.to_cout("")


def test_pointee_and_ptr_owner():
    m.to_cout("")
    obj = m.pointee()
    assert obj.get_int() == 213
    owner = m.ptr_owner(obj)
    with pytest.raises(RuntimeError) as exc_info:
        obj.get_int()
    assert str(exc_info.value) == "Invalid object instance"
    assert owner.is_owner()
    m.to_cout("before give up")
    reclaimed = owner.give_up_ownership_via_shared_ptr()
    m.to_cout("after give up")
    assert not owner.is_owner()
    # assert reclaimed.get_int() == 213
    del reclaimed
    m.to_cout("after del")
    m.to_cout("3")
    m.to_cout("")


def test_cpp_pattern():
    m.to_cout("")
    res = m.cpp_pattern()
    assert res == 111111
    m.to_cout("")
