# -*- coding: utf-8 -*-
import pytest

import env
from pybind11_tests import const_name as m


@pytest.mark.parametrize("func", (m.const_name_tests, m.underscore_tests))
@pytest.mark.parametrize(
    "selector, expected",
    enumerate(
        (
            "",
            "A",
            "Bd",
            "Cef",
            "%",
            "%",
            "T1",
            "U2",
            "D1",
            "E2",
            "KeepAtEnd",
        )
    ),
)
def test_const_name(func, selector, expected):
    if isinstance(func, type(u"") if env.PY2 else str):
        pytest.skip(func)
    text = func(selector)
    assert text == expected
