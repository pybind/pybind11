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
    from pybind11_tests import (MyException, MyException5, MyException5_1,
                                throws1, throws2, throws3, throws4, throws5, throws5_1,
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

    # Can we handle a helper-declared exception?
    with pytest.raises(MyException5) as excinfo:
        throws5()
    assert msg(excinfo.value) == "this is a helper-defined translated exception"

    # Exception subclassing:
    with pytest.raises(MyException5) as excinfo:
        throws5_1()
    assert msg(excinfo.value) == "MyException5 subclass"
    assert isinstance(excinfo.value, MyException5_1)

    with pytest.raises(MyException5_1) as excinfo:
        throws5_1()
    assert msg(excinfo.value) == "MyException5 subclass"

    with pytest.raises(MyException5) as excinfo:
        try:
            throws5()
        except MyException5_1:
            raise RuntimeError("Exception error: caught child from parent")
    assert msg(excinfo.value) == "this is a helper-defined translated exception"
