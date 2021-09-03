# -*- coding: utf-8 -*-

import threading

from pybind11_tests import thread as m


def test_implicit_conversion():
    def loop(count):
        for i in range(count):
            m.test(i)

    a = threading.Thread(target=loop, args=(10,))
    b = threading.Thread(target=loop, args=(10,))
    c = threading.Thread(target=loop, args=(10,))
    for x in [a, b, c]:
        x.start()
    for x in [c, b, a]:
        x.join()


def test_implicit_conversion_no_gil():
    def loop(count):
        for i in range(count):
            m.test_no_gil(i)

    a = threading.Thread(target=loop, args=(10,))
    b = threading.Thread(target=loop, args=(10,))
    c = threading.Thread(target=loop, args=(10,))
    for x in [a, b, c]:
        x.start()
    for x in [c, b, a]:
        x.join()
