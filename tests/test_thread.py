# -*- coding: utf-8 -*-
import pytest
from pybind11_tests import test_threading as m
from pybind11_tests import iostream
import time

def test_threading():
  with iostream.ostream_redirect(stdout=True, stderr=False):
    # start some threads
    threads = []

    # start some threads
    for j in range(20):
      threads.append( m.TestThread() )

    # give the threads some time to fail
    threads[0].sleep()

    # stop all the threads
    for t in threads:
      t.stop()

    for t in threads:
      t.join()

  # if a thread segfaults, we don't get here
  assert True

