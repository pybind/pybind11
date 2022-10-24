import cross_module_exception_odr_1 as m
import cross_module_exception_odr_2 as m2
import pytest


def test_raise_evolving():
    with pytest.raises(RuntimeError, match="v1:t1"):
        m.raise_evolving("t1")


def test_raise_evolving_from_module_2():
    cap = m2.get_raise_evolving_from_module_2_capsule()
    with pytest.raises(RuntimeError, match="v2"):
        m.raise_evolving_from_module_2(cap)
