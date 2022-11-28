from pybind11_tests import return_value_policy_override as m


def test_return_value():
    assert m.return_value_with_default_policy() == "move"
    assert m.return_value_with_policy_copy() == "move"
    assert m.return_value_with_policy_clif_automatic() == "_clif_automatic"


def test_return_pointer():
    assert m.return_pointer_with_default_policy() == "automatic"
    assert m.return_pointer_with_policy_move() == "move"
    assert m.return_pointer_with_policy_clif_automatic() == "_clif_automatic"
