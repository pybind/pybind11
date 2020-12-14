# -*- coding: utf-8 -*-
import pytest

import env  # noqa: F401

from pybind11_tests import const_ref_caster as m


def test_takes():
	assert m.takes(m.ConstRefCasted())

	assert m.takes_ptr(m.ConstRefCasted())
	assert m.takes_ref(m.ConstRefCasted())
	assert m.takes_ref_wrap(m.ConstRefCasted())

	assert m.takes_const_ptr(m.ConstRefCasted())
	assert m.takes_const_ref(m.ConstRefCasted())
	assert m.takes_const_ref_wrap(m.ConstRefCasted())
