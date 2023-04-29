import pytest

import pybind11_tests.class_sh_trampoline_self_life_support as m


class PyBig5(m.Big5):
    pass


def test_m_big5():
    obj = m.Big5("Seed")
    assert obj.history == "Seed"
    o1, o2 = m.action(obj, 0)
    assert o1 is not obj
    assert o1.history == "Seed"
    with pytest.raises(ValueError) as excinfo:
        _ = obj.history
    assert "Python instance was disowned" in str(excinfo.value)
    assert o2 is None


@pytest.mark.parametrize(
    ("action_id", "expected_history"),
    [
        (0, "Seed_CpCtor"),
        (1, "Seed_MvCtor"),
        (2, "Seed_OpEqLv"),
        (3, "Seed_OpEqRv"),
    ],
)
def test_py_big5(action_id, expected_history):
    obj = PyBig5("Seed")
    assert obj.history == "Seed"
    o1, o2 = m.action(obj, action_id)
    assert o1 is obj
    assert o2.history == expected_history
