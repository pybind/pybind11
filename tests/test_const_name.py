# -*- coding: utf-8 -*-
import itertools

import pytest

from pybind11_tests import const_name as m


@pytest.mark.parametrize(
    "func, selector_expected",
    itertools.product(
        (m.const_name_tests, m.underscore_tests),
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
    ),
)
def test_const_name(func, selector_expected):
    selector, expected = selector_expected
    if func is None:
        pytest.skip("PYBIND11_DETAIL_UNDERSCORE_BACKWARD_COMPATIBILITY not defined.")
    text = func(selector)
    assert text == expected
