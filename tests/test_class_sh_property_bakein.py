from __future__ import annotations

import pytest

from pybind11_tests import class_sh_property_bakein as m

if not m.defined_PYBIND11_HAVE_INTERNALS_WITH_SMART_HOLDER_SUPPORT:
    pytest.skip("smart_holder not available.", allow_module_level=True)


def test_readonly_char6_member():
    obj = m.WithCharArrayMember()
    assert obj.char6_member == "Char6"


def test_readonly_const_char_ptr_member():
    obj = m.WithConstCharPtrMember()
    assert obj.const_char_ptr_member == "ConstChar*"
