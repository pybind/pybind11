import pytest

from pybind11_tests import utility as m


@pytest.mark.skipif(not hasattr(m, "has_optional"), reason='no <optional>')
def test_optional():
    assert m.double_or_zero(None) == 0
    assert m.double_or_zero(42) == 84
    pytest.raises(TypeError, m.double_or_zero, 'foo')

    assert m.half_or_none(0) is None
    assert m.half_or_none(42) == 21
    pytest.raises(TypeError, m.half_or_none, 'foo')

    assert m.test_nullopt() == 42
    assert m.test_nullopt(None) == 42
    assert m.test_nullopt(42) == 42
    assert m.test_nullopt(43) == 43

    assert m.test_no_assign() == 42
    assert m.test_no_assign(None) == 42
    assert m.test_no_assign(m.NoAssign(43)) == 43
    pytest.raises(TypeError, m.test_no_assign, 43)

    assert m.nodefer_none_optional(None)


@pytest.mark.skipif(not hasattr(m, "has_exp_optional"), reason='no <experimental/optional>')
def test_exp_optional():
    assert m.double_or_zero_exp(None) == 0
    assert m.double_or_zero_exp(42) == 84
    pytest.raises(TypeError, m.double_or_zero_exp, 'foo')

    assert m.half_or_none_exp(0) is None
    assert m.half_or_none_exp(42) == 21
    pytest.raises(TypeError, m.half_or_none_exp, 'foo')

    assert m.test_nullopt_exp() == 42
    assert m.test_nullopt_exp(None) == 42
    assert m.test_nullopt_exp(42) == 42
    assert m.test_nullopt_exp(43) == 43

    assert m.test_no_assign_exp() == 42
    assert m.test_no_assign_exp(None) == 42
    assert m.test_no_assign_exp(m.NoAssign(43)) == 43
    pytest.raises(TypeError, m.test_no_assign_exp, 43)


@pytest.mark.skipif(not hasattr(m, "load_variant"), reason='no <variant>')
def test_variant(doc):
    assert m.load_variant(1) == "int"
    assert m.load_variant("1") == "std::string"
    assert m.load_variant(1.0) == "double"
    assert m.load_variant(None) == "std::nullptr_t"

    assert m.load_variant_2pass(1) == "int"
    assert m.load_variant_2pass(1.0) == "double"

    assert m.cast_variant() == (5, "Hello")

    assert doc(m.load_variant) == "load_variant(arg0: Union[int, str, float, None]) -> str"
