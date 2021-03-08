# -*- coding: utf-8 -*-
import re

import pytest

from pybind11_tests import type_caster_bare_interface as m


@pytest.mark.parametrize(
    "rtrn_f, expected",
    [
        (m.rtrn_valu, "cast_rref:valu_CpCtor"),
        (m.rtrn_rref, "cast_rref:rref"),
        (m.rtrn_cref, "cast_cref:cref"),
        (m.rtrn_mref, "cast_mref:mref"),
        (m.rtrn_cptr, "cast_cptr:cptr"),
        (m.rtrn_mptr, "cast_mptr:mptr"),
    ],
)
def test_cast(rtrn_f, expected):
    assert re.match(expected, rtrn_f())


@pytest.mark.parametrize(
    "pass_f, expected",
    [
        (m.pass_valu, "pass_valu:rref_MvCtor"),
        (m.pass_rref, "pass_rref:rref"),
        (m.pass_cref, "pass_cref:cref"),
        (m.pass_mref, "pass_mref:mref"),
        (m.pass_cptr, "pass_cptr:cptr"),
        (m.pass_mptr, "pass_mptr:mptr"),
    ],
)
def test_operator(pass_f, expected):
    assert re.match(expected, pass_f(None))
