from __future__ import annotations

import pytest

asyncio = pytest.importorskip("asyncio")
m = pytest.importorskip("pybind11_tests.const_module")


def test_const_smart_ptr():
    cons = m.Data("my_name")
    assert cons.name == "my_name"
