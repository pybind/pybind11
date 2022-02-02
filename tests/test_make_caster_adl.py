# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import make_caster_adl as m


def test_mock_casters():
    assert m.have_a_ns_num() == 101
    assert m.global_ns_num() == 202
    assert m.unnamed_ns_num() == 303


def test_minimal_real_caster():
    assert m.mrc_return() == 1505
    assert m.mrc_arg(u"ignored") == 2404
    with pytest.raises(TypeError) as excinfo:
        m.mrc_arg(None)
    assert "(arg0: mrc_ns::type_mrc) -> int" in str(excinfo.value)
