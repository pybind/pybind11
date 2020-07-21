# -*- coding: utf-8 -*-
import os
import pytest
from pybind11_tests import eval_ as m


def test_evals(capture):
    with capture:
        assert m.test_eval_statements()
    assert capture == "Hello World!"

    assert m.test_eval()
    assert m.test_eval_single_statement()

    assert m.test_eval_failure()


@pytest.unsupported_on_pypy3
def test_eval_file():
    filename = os.path.join(os.path.dirname(__file__), "test_eval_call.py")
    if isinstance(filename, bytes):  # true for Python 2 only
        filename = filename.decode()  # effectively six.ensure_text()
    assert m.test_eval_file(filename)

    assert m.test_eval_file_failure()
