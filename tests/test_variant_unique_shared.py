# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import variant_unique_shared as m


def test_default_constructed():
    v = m.vptr_holder_double()
    assert v.ownership_type() == 0
    assert v.get_value() == -1


def test_from_raw():
    v = m.from_raw()
    assert v.ownership_type() == 0
    assert v.get_value() == 3


def test_from_unique():
    v = m.from_unique()
    assert v.ownership_type() == 0
    assert v.get_value() == 5


def test_from_shared():
    v = m.from_shared()
    assert v.ownership_type() == 1
    assert v.get_value() == 7


def test_promotion_to_shared():
    v = m.from_raw()
    v.get_unique()
    assert v.ownership_type() == 0
    v.get_shared()  # Promotion to shared_ptr.
    assert v.ownership_type() == 1
    v.get_shared()  # Existing shared_ptr.
    with pytest.raises(RuntimeError) as exc_info:
        v.get_unique()
    assert str(exc_info.value) == "get_unique failure."
    v.get_shared()  # Still works.


def test_shared_from_birth():
    v = m.from_shared()
    assert v.ownership_type() == 1
    with pytest.raises(RuntimeError) as exc_info:
        v.get_unique()
    assert str(exc_info.value) == "get_unique failure."
    v.get_shared()  # Still works.
