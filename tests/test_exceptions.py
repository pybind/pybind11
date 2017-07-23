import pytest

from pybind11_tests import exceptions as m


def test_std_exception(msg):
    with pytest.raises(RuntimeError) as excinfo:
        m.throw_std_exception()
    assert msg(excinfo.value) == "This exception was intentionally thrown."


def test_error_already_set(msg):
    with pytest.raises(RuntimeError) as excinfo:
        m.throw_already_set(False)
    assert msg(excinfo.value) == "Unknown internal error occurred"

    with pytest.raises(ValueError) as excinfo:
        m.throw_already_set(True)
    assert msg(excinfo.value) == "foo"


def test_python_call_in_catch():
    d = {}
    assert m.python_call_in_destructor(d) is True
    assert d["good"] is True


def test_exception_matches():
    m.exception_matches()


def test_custom(msg):
    # Can we catch a MyException?
    with pytest.raises(m.MyException) as excinfo:
        m.throws1()
    assert msg(excinfo.value) == "this error should go to a custom type"

    # Can we translate to standard Python exceptions?
    with pytest.raises(RuntimeError) as excinfo:
        m.throws2()
    assert msg(excinfo.value) == "this error should go to a standard Python exception"

    # Can we handle unknown exceptions?
    with pytest.raises(RuntimeError) as excinfo:
        m.throws3()
    assert msg(excinfo.value) == "Caught an unknown exception!"

    # Can we delegate to another handler by rethrowing?
    with pytest.raises(m.MyException) as excinfo:
        m.throws4()
    assert msg(excinfo.value) == "this error is rethrown"

    # Can we fall-through to the default handler?
    with pytest.raises(RuntimeError) as excinfo:
        m.throws_logic_error()
    assert msg(excinfo.value) == "this error should fall through to the standard handler"

    # Can we handle a helper-declared exception?
    with pytest.raises(m.MyException5) as excinfo:
        m.throws5()
    assert msg(excinfo.value) == "this is a helper-defined translated exception"

    # Exception subclassing:
    with pytest.raises(m.MyException5) as excinfo:
        m.throws5_1()
    assert msg(excinfo.value) == "MyException5 subclass"
    assert isinstance(excinfo.value, m.MyException5_1)

    with pytest.raises(m.MyException5_1) as excinfo:
        m.throws5_1()
    assert msg(excinfo.value) == "MyException5 subclass"

    with pytest.raises(m.MyException5) as excinfo:
        try:
            m.throws5()
        except m.MyException5_1:
            raise RuntimeError("Exception error: caught child from parent")
    assert msg(excinfo.value) == "this is a helper-defined translated exception"
