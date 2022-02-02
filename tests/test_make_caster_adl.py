# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import make_caster_adl as m


def test_mock_caster():
    assert m.num_mock() == 101


def test_minimal_real_caster():
    assert m.obj_mrc_return() == 1404
    assert m.obj_mrc_arg(u"ignored") == 2303
    with pytest.raises(TypeError) as excinfo:
        m.obj_mrc_arg(None)
    assert "(arg0: adl_mrc::type_mrc) -> int" in str(excinfo.value)
