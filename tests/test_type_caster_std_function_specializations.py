from pybind11_tests import type_caster_std_function_specializations as m


def test_callback():
    def func():
        return m.SpecialReturn()

    assert func().value == 99
    assert m.call_callback_with_special_return(func).value == 199
