import cross_module_exception_odr_2 as m
import pytest


def test_raise_evolving():
    with pytest.raises(RuntimeError, match="v2"):
        m.raise_evolving()
