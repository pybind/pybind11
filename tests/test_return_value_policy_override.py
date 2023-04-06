import re

import pytest

from pybind11_tests import return_value_policy_override as m


def test_return_value():
    assert m.return_value_with_default_policy() == "move"
    assert m.return_value_with_policy_copy() == "move"
    assert m.return_value_with_policy_clif_automatic() == "_clif_automatic"


def test_return_pointer():
    assert m.return_pointer_with_default_policy() == "automatic"
    assert m.return_pointer_with_policy_move() == "move"
    assert m.return_pointer_with_policy_clif_automatic() == "_clif_automatic"


def test_persistent_holder():
    h = m.data_fields_holder(2)
    assert h.vec_at(0).value == 13
    assert h.vec_at(1).value == 24
    assert h.vec_at_const_ptr(0).value == 13
    assert h.vec_at_const_ptr(1).value == 24


def test_temporary_holder():
    data_field = m.data_fields_holder(2).vec_at(1)
    assert data_field.value == 24
    data_field = m.data_fields_holder(2).vec_at_const_ptr(1)
    assert data_field.value == 24


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        (m.return_value, "value(_MvCtor)*_MvCtor"),
        (m.return_pointer, "pointer"),
        (m.return_const_pointer, "const_pointer_CpCtor"),
        (m.return_reference, "reference_MvCtor"),
        (m.return_const_reference, "const_reference_CpCtor"),
        (m.return_unique_pointer, "unique_pointer"),
        (m.return_shared_pointer, "shared_pointer"),
        (m.return_value_nocopy, "value_nocopy(_MvCtor)*_MvCtor"),
        (m.return_pointer_nocopy, "pointer_nocopy"),
        (m.return_reference_nocopy, "reference_nocopy_MvCtor"),
        (m.return_unique_pointer_nocopy, "unique_pointer_nocopy"),
        (m.return_shared_pointer_nocopy, "shared_pointer_nocopy"),
        (m.return_pointer_nomove, "pointer_nomove"),
        (m.return_const_pointer_nomove, "const_pointer_nomove_CpCtor"),
        (m.return_reference_nomove, "reference_nomove_CpCtor"),
        (m.return_const_reference_nomove, "const_reference_nomove_CpCtor"),
        (m.return_unique_pointer_nomove, "unique_pointer_nomove"),
        (m.return_shared_pointer_nomove, "shared_pointer_nomove"),
        (m.return_pointer_nocopy_nomove, "pointer_nocopy_nomove"),
        (m.return_unique_pointer_nocopy_nomove, "unique_pointer_nocopy_nomove"),
        (m.return_shared_pointer_nocopy_nomove, "shared_pointer_nocopy_nomove"),
    ],
)
def test_clif_automatic_return_value_policy_override(func, expected):
    assert re.match(expected, func().mtxt)
