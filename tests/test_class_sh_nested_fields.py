# -*- coding: utf-8 -*-
import pytest

from pybind11_tests import class_sh_nested_fields as m


def test_bc(msg):
    m.to_cout("")
    m.to_cout("HELLO")
    b = m.BB()
    m.to_cout("have b")
    c = b.c
    m.to_cout("have c")
    with pytest.raises(ValueError) as excinfo:
        m.ConsumeCC(c)
    assert (
        msg(excinfo.value) == "Cannot disown non-owning holder (loaded_as_unique_ptr)."
    )


def test_abc():
    m.to_cout("")
    m.to_cout("LOOOK TEST_ABC")
    a = m.AA()
    m.to_cout("LOOOK have a")
    b = a.b
    m.to_cout("LOOOK have b")
    c = b.c
    m.to_cout("LOOOK have c")
    assert c.i == 13
    m.to_cout("LOOOK c.i reset")
    m.ConsumeAA(a)  # Without this it works.
    m.to_cout("LOOOK ConsumeAA done")
    assert (
        c.i == 13
    )  # AddressSanitizer: heap-use-after-free pybind11.h:235:52 in operator()
    # 233 /* Perform the function call */
    # 234 handle result = cast_out::cast(
    # 235     std::move(args_converter).template call<Return, Guard>(cap->f), policy, call.parent);
    m.to_cout("LOOOK c.i equality done")
