# -*- coding: utf-8 -*-
import pytest

import concurrent.futures
import env  # noqa: F401
from pybind11_tests import thread as m


def method(s):
   return m.method(s)

def method_no_gil(s):
   return m.method_no_gil(s)


def test_message():
    inputs = [ '%d' % i for i in range(20,30) ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(method, inputs))
    results.sort()
    for i in range(len(results)):
        assert results[i] == ('%s' % (i + 20))


def test_message_no_gil():
    inputs = [ '%d' % i for i in range(20,30) ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(method_no_gil, inputs))
    results.sort()
    for i in range(len(results)):
        assert results[i] == ('%s' % (i + 20))
