from __future__ import annotations

from pybind11_tests import class_sh_property_bakein as m


def test_readonly_char6_member():
    obj = m.WithCharArrayMember()
    assert obj.char6_member == "Char6"


def test_readonly_const_char_ptr_member():
    obj = m.WithConstCharPtrMember()
    assert obj.const_char_ptr_member == "ConstChar*"
