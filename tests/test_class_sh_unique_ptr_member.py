# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import class_sh_unique_ptr_member as m


def test_make_unique_pointee():
    obj = m.make_unique_pointee()
    assert obj.get_int() == 213


@pytest.mark.parametrize(
    "give_up_ownership_via",
    ["give_up_ownership_via_unique_ptr", "give_up_ownership_via_shared_ptr"],
)
def test_pointee_and_ptr_owner(give_up_ownership_via):
    obj = m.pointee()
    assert obj.get_int() == 213
    owner = m.ptr_owner(obj)
    with pytest.raises(RuntimeError) as exc_info:
        obj.get_int()
    assert (
        str(exc_info.value)
        == "Missing value for wrapped C++ type: Python instance is uninitialized or was disowned."
    )
    assert owner.is_owner()
    reclaimed = getattr(owner, give_up_ownership_via)()
    assert not owner.is_owner()
    assert reclaimed.get_int() == 213
