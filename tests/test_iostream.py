import pytest
from pybind11_tests import iostream as m


def test_captured(capture):
    with capture:
        m.captured_output("I've been redirected to Python, I hope!")
    assert capture == "I've been redirected to Python, I hope!"


def test_not_captured(capture):
    with capture:
        m.raw_output(" <OK> ")
    assert capture == ""


def test_series_captured(capture):
    with capture:
        m.captured_output("a")
        m.captured_output("b")
    assert capture == "ab"


def test_multi_captured(capture):
    with capture:
        m.captured_output("a")
        m.raw_output(" <OK> ")
        m.captured_output("b")
    assert capture == "ab"


# capfd seems to be unstable on pypy
@pytest.unsupported_on_pypy
def test_successful(capfd):
    m.raw_output("I've been output to cout, I hope!")
    stdout, stderr = capfd.readouterr()
    assert stdout == "I've been output to cout, I hope!"


def test_dual(capsys):
    m.captured_dual("a", "b")
    stdout, stderr = capsys.readouterr()
    assert stdout == "a"
    assert stderr == "b"
