from __future__ import annotations

import pytest

from pybind11_tests import class_sh_unique_ptr_custom_deleter as m

if not m.defined_PYBIND11_SMART_HOLDER_ENABLED:
    pytest.skip("smart_holder not available.", allow_module_level=True)


def test_create():
    pet = m.create("abc")
    assert pet.name == "abc"
