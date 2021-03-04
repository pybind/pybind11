# -*- coding: utf-8 -*-
import os

import pytest

import env  # noqa: F401

from pybind11_tests import eval_ as m


def test_evals(capture):
    with capture:
        assert m.test_eval_statements()
    assert capture == "Hello World!"

    assert m.test_eval()
    assert m.test_eval_single_statement()

    assert m.test_eval_failure()


@pytest.mark.xfail("env.PYPY and not env.PY2", raises=RuntimeError)
def test_eval_file():
    filename = os.path.join(os.path.dirname(__file__), "test_eval_call.py")
    assert m.test_eval_file(filename)

    assert m.test_eval_file_failure()


def test_eval_empty_globals():
    assert "__builtins__" in m.eval_empty_globals(None)

    g = {}
    assert "__builtins__" in m.eval_empty_globals(g)
    assert "__builtins__" in g
