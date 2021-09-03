# -*- coding: utf-8 -*-
import concurrent.futures

import pytest

import env  # noqa: F401
from pybind11_tests import thread as m


def test_implicit_conversion():
    inputs = [i for i in range(20,30) ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(m.test, inputs)


def test_implicit_conversion_no_gil():
    inputs = [i for i in range(20,30) ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(m.test_no_gil, inputs)
