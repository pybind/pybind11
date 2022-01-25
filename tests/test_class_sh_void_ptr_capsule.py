# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import class_sh_void_ptr_capsule as m


@pytest.mark.parametrize(
    "ctor, caller, expected, capsule_generated",
    [
        (m.Valid, m.get_from_valid_capsule, 101, True),
        (m.NoConversion, m.get_from_no_conversion_capsule, 102, False),
        (m.NoCapsuleReturned, m.get_from_no_capsule_returned, 103, True),
        (m.AsAnotherObject, m.get_from_valid_capsule, 104, True),
    ],
)
def test_as_void_ptr_capsule(ctor, caller, expected, capsule_generated):
    obj = ctor()
    assert caller(obj) == expected
    assert obj.capsule_generated == capsule_generated


@pytest.mark.parametrize(
    "ctor, caller, pointer_type, capsule_generated",
    [
        (m.AsAnotherObject, m.get_from_shared_ptr_valid_capsule, "shared_ptr", True),
        (m.AsAnotherObject, m.get_from_unique_ptr_valid_capsule, "unique_ptr", True),
    ],
)
def test_as_void_ptr_capsule_unsupported(ctor, caller, pointer_type, capsule_generated):
    obj = ctor()
    with pytest.raises(RuntimeError) as excinfo:
        caller(obj)
    assert pointer_type in str(excinfo.value)
    assert obj.capsule_generated == capsule_generated
