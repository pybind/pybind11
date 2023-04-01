import pytest

from pybind11_tests import type_caster_pyobject_ptr as m


def test_cast_from_pyobject_ptr():
    assert m.cast_from_pyobject_ptr() == 6758


def test_cast_to_pyobject_ptr():
    assert m.cast_to_pyobject_ptr(())
    assert not m.cast_to_pyobject_ptr({})


def test_return_pyobject_ptr():
    assert m.return_pyobject_ptr() == 2314


def test_pass_pyobject_ptr():
    assert m.pass_pyobject_ptr(())
    assert not m.pass_pyobject_ptr({})


@pytest.mark.parametrize(
    "call_callback",
    [
        m.call_callback_with_object_return,
        m.call_callback_with_handle_return,
        m.call_callback_with_pyobject_ptr_return,
    ],
)
def test_call_callback_with_object_return(call_callback):
    def cb(mode):
        if mode == 0:
            return 10
        if mode == 1:
            return "One"
        raise NotImplementedError(f"Unknown mode: {mode}")

    assert call_callback(cb, 0) == 10
    assert call_callback(cb, 1) == "One"
    with pytest.raises(NotImplementedError, match="Unknown mode: 2"):
        call_callback(cb, 2)


def test_call_callback_with_pyobject_ptr_arg():
    def cb(obj):
        return isinstance(obj, tuple)

    assert m.call_callback_with_pyobject_ptr_arg(cb, ())
    assert not m.call_callback_with_pyobject_ptr_arg(cb, {})


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
