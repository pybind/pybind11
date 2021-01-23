# -*- coding: utf-8 -*-
import pytest

import classh_module_local_0 as m0
import classh_module_local_1 as m1
import classh_module_local_2 as m2


def test_cross_module_get_msg():
    b1 = m1.bottle("A")
    assert b1.tag() == 1
    b2 = m2.bottle("B")
    assert b2.tag() == 2
    assert m1.get_msg(b1) == "A"
    assert m2.get_msg(b2) == "B"
    assert m1.get_msg(b2) == "B"
    assert m2.get_msg(b1) == "A"
    assert m0.get_msg(b1) == "A"
    assert m0.get_msg(b2) == "B"


def test_m0_make_bottle():
    with pytest.raises(TypeError) as exc_info:
        m0.make_bottle()
    assert str(exc_info.value).startswith(
        "Unable to convert function return value to a Python type!"
    )
