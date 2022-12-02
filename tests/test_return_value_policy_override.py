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


@pytest.mark.parametrize(
    "func, expected",
    [
        (m.return_object_value_with_policy_clif_automatic, "value_MvCtor"),
        (m.return_object_pointer_with_policy_clif_automatic, "pointer"),
        (
            m.return_object_const_pointer_with_policy_clif_automatic,
            "const_pointer_CpCtor",
        ),
        (m.return_object_reference_with_policy_clif_automatic, "reference_MvCtor"),
        (
            m.return_object_const_reference_with_policy_clif_automatic,
            "const_reference_CpCtor",
        ),
        (m.return_object_unique_ptr_with_policy_clif_automatic, "unique_pointer"),
        (m.return_object_shared_ptr_with_policy_clif_automatic, "shared_pointer"),
        (
            m.return_nocopy_reference_with_policy_clif_automatic,
            "reference_nocopy_MvCtor",
        ),
    ],
)
def test_clif_automatic_return_value_policy_override(func, expected):
    assert func().mtxt == expected
