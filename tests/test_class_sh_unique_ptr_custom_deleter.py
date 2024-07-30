from __future__ import annotations

import pytest

from pybind11_tests import class_sh_unique_ptr_custom_deleter as m

if not m.defined_PYBIND11_HAVE_INTERNALS_WITH_SMART_HOLDER_SUPPORT:
    pytest.skip("smart_holder not available.", allow_module_level=True)


def test_create():
    pet = m.create("abc")
    assert pet.name == "abc"
