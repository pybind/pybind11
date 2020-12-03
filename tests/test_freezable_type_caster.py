# -*- coding: utf-8 -*-
import pytest

import env  # noqa: F401

from pybind11_tests import test_freezable_type_caster as m


def test_make():
    assert m.make().x == 1


def test_take():
    c = m.make()
    assert m.take(c) == 2
    assert m.take(c) == 2
    assert m.take_ptr(c) == 2
    assert m.take_ptr(c) == 3
    assert m.take_ref(c) == 4
    assert m.take_ref(c) == 5
    assert m.take_wrap(c) == 6
    assert m.take_wrap(c) == 7

    assert m.take_const_ptr(c) == 70
    assert m.take_const_ref(c) == 70
    assert m.take_const_wrap(c) == 70


def test_get():
    """The get calls all return a variant of the same shared pointer."""
    m.reset(1)
    assert m.get().x == 1
    v = m.get_ptr()
    assert v.x == 1
    v.x = 10
    assert m.get_ptr().x == 10
    v.x = 11
    assert m.get_ref().x == 11
    v.x = 12
    assert m.get_wrap().x == 12
    v.x = 13
    assert m.get().x == 13


def test_get_const():
    m.reset(1)
    v = m.get_const_ptr()
    assert v.x == 2
    assert m.get_const_ref().x == 3
    assert m.get_const_wrap().x == 4
    assert m.get_const_ptr().x == 5
    assert m.get_const_ref().x == 6
    assert m.get_const_wrap().x == 7
    assert v.x == 7

    with pytest.raises(ValueError):
        v.x = 1  # immutable


def test_roundtrip():
    c = m.make()
    assert c.x == 1
    assert m.roundtrip(c).x == 2
    assert m.roundtrip(c).x == 2

    assert m.roundtrip_ptr(c).x == 2
    assert m.roundtrip_ref(c).x == 3
    assert m.roundtrip_wrap(c).x == 4


def test_roundtrip_const():
    m.reset(1)

    # NOTE: each call to get_const_ref increments x.
    # by reference, the const ref is not converted.
    with pytest.raises(TypeError):
        m.roundtrip_ref(m.get_const_ref())

    with pytest.raises(TypeError):
        m.roundtrip_ptr(m.get_const_ref())

    with pytest.raises(TypeError):
        m.roundtrip_wrap(m.get_const_ref())

    # by value, so the const ref is converted.
    assert m.roundtrip(m.get_const_ref()).x == 6  # x + 1

    # by const ref, so the conversion is ok.
    assert m.roundtrip_const_ref(m.get_const_ref()).x == 6
