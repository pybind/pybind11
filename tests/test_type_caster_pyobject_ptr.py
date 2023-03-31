import pytest

from pybind11_tests import type_caster_pyobject_ptr as m


def test_cast_from_PyObject_ptr():
    assert m.cast_from_PyObject_ptr() == 6758


def test_cast_to_PyObject_ptr():
    assert m.cast_to_PyObject_ptr(())
    assert not m.cast_to_PyObject_ptr({})


def test_return_PyObject_ptr():
    assert m.return_PyObject_ptr() == 2314


def test_pass_PyObject_ptr():
    assert m.pass_PyObject_ptr(())
    assert not m.pass_PyObject_ptr({})


@pytest.mark.parametrize(
    "call_callback",
    [
        m.call_callback_with_object_return,
        m.call_callback_with_handle_return,
        m.call_callback_with_PyObject_ptr_return,
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


def test_call_callback_with_PyObject_ptr_arg():
    def cb(obj):
        return isinstance(obj, tuple)

    assert m.call_callback_with_PyObject_ptr_arg(cb, ())
    assert not m.call_callback_with_PyObject_ptr_arg(cb, {})
