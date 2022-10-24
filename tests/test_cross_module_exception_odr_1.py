import cross_module_exception_odr_1 as m
import pytest


def test_raise_evolving():
    with pytest.raises(RuntimeError, match="v1:t1"):
        m.raise_evolving("t1")
