import pytest


def test_error_already_set(msg):
    from pybind11_tests import throw_already_set

    with pytest.raises(RuntimeError) as excinfo:
        throw_already_set(False)
    assert msg(excinfo.value) == "Unknown internal error occurred"

    with pytest.raises(ValueError) as excinfo:
        throw_already_set(True)
    assert msg(excinfo.value) == "foo"


def test_python_call_in_catch():
    from pybind11_tests import python_call_in_destructor

    d = {}
    assert python_call_in_destructor(d) is True
    assert d["good"] is True


def test_custom(msg):
    from pybind11_tests import (MyException, throws1, throws2, throws3, throws4,
                                throws_logic_error)

    # Can we catch a MyException?"
    with pytest.raises(MyException) as excinfo:
        throws1()
    assert msg(excinfo.value) == "this error should go to a custom type"

    # Can we translate to standard Python exceptions?
    with pytest.raises(RuntimeError) as excinfo:
        throws2()
    assert msg(excinfo.value) == "this error should go to a standard Python exception"

    # Can we handle unknown exceptions?
    with pytest.raises(RuntimeError) as excinfo:
        throws3()
    assert msg(excinfo.value) == "Caught an unknown exception!"

    # Can we delegate to another handler by rethrowing?
    with pytest.raises(MyException) as excinfo:
        throws4()
    assert msg(excinfo.value) == "this error is rethrown"

    # "Can we fall-through to the default handler?"
    with pytest.raises(RuntimeError) as excinfo:
        throws_logic_error()
    assert msg(excinfo.value) == "this error should fall through to the standard handler"
