import pytest

from pybind11_tests import type_caster_pyobject_ptr as m


# For use as a temporary user-defined object, to maximize sensitivity of the tests below.
class ValueHolder:
    def __init__(self, value):
        self.value = value


def test_cast_from_pyobject_ptr():
    assert m.cast_from_pyobject_ptr() == 6758


def test_cast_to_pyobject_ptr():
    assert m.cast_to_pyobject_ptr(ValueHolder(24)) == 76


def test_return_pyobject_ptr():
    assert m.return_pyobject_ptr() == 2314


def test_pass_pyobject_ptr():
    assert m.pass_pyobject_ptr(ValueHolder(82)) == 118


@pytest.mark.parametrize(
    "call_callback",
    [
        m.call_callback_with_object_return,
        m.call_callback_with_pyobject_ptr_return,
    ],
)
def test_call_callback_with_object_return(call_callback):
    def cb(value):
        if value < 0:
            raise ValueError("Raised from cb")
        return ValueHolder(1000 - value)

    assert call_callback(cb, 287).value == 713

    with pytest.raises(ValueError, match="^Raised from cb$"):
        call_callback(cb, -1)


def test_call_callback_with_pyobject_ptr_arg():
    def cb(obj):
        return 300 - obj.value

    assert m.call_callback_with_pyobject_ptr_arg(cb, ValueHolder(39)) == 261


@pytest.mark.parametrize("set_error", [True, False])
def test_cast_to_python_nullptr(set_error):
    expected = {
        True: r"^Reflective of healthy error handling\.$",
        False: (
            r"^Internal error: pybind11::error_already_set called "
            r"while Python error indicator not set\.$"
        ),
    }[set_error]
    with pytest.raises(RuntimeError, match=expected):
        m.cast_to_pyobject_ptr_nullptr(set_error)


def test_cast_to_python_non_nullptr_with_error_set():
    with pytest.raises(SystemError) as excinfo:
        m.cast_to_pyobject_ptr_non_nullptr_with_error_set()
    assert str(excinfo.value) == "src != nullptr but PyErr_Occurred()"
    assert str(excinfo.value.__cause__) == "Reflective of unhealthy error handling."
