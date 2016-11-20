import pytest


def test_lacking_copy_ctor():
    from pybind11_tests import lacking_copy_ctor
    with pytest.raises(RuntimeError) as excinfo:
        lacking_copy_ctor.get_one()
    assert "the object is non-copyable!" in str(excinfo.value)


def test_lacking_move_ctor():
    from pybind11_tests import lacking_move_ctor
    with pytest.raises(RuntimeError) as excinfo:
        lacking_move_ctor.get_one()
    assert "the object is neither movable nor copyable!" in str(excinfo.value)
