import named_namespace_a as m
import pytest


def test_inspect():
    assert m.name == "NA"
    reg = m.std_type_index_registry_dump()
    if len(reg) == 1:
        assert tuple(sorted(reg[0][1])) == ("NA", "NB")
        pytest.skip("std::type_index-EQ-GOOD")
    if len(reg) == 2:
        assert reg[0][0] == reg[1][0]
        assert tuple(sorted(reg[0][1] + reg[1][1])) == ("NA", "NB")
        pytest.skip("std::type_index-NE-BAD")
    assert reg is None  # Sure to fail.
