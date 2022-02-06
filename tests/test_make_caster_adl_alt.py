# -*- coding: utf-8 -*-

from pybind11_tests import make_caster_adl_alt as m


def test_mock_casters():
    assert m.have_a_ns_num() == 121
