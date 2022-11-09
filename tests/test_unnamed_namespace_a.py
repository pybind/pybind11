import pytest

from pybind11_tests import unnamed_namespace_a as m


def test_inspect():
    assert m.name == "UA"
    reg = m.std_type_index_registry_dump()
    if len(reg) == 1:
        assert tuple(sorted(reg[0][1])) == ("UA", "UB")
        pytest.skip("std::type_index-EQ-BAD")
    if len(reg) == 2:
        assert tuple(sorted([reg[0][1][0], reg[1][1][0]])) == ("UA", "UB")
        pytest.skip("std::type_index-NE-GOOD")
    assert reg is None  # Sure to fail.
