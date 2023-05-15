from __future__ import annotations

import sys

import pytest

if sys.platform.startswith("emscripten"):
    pytest.skip(
        "Test not implemented from single wheel on Pyodide", allow_module_level=True
    )

import trampoline_module


def func():
    class Test(trampoline_module.test_override_cache_helper):
        def func(self):
            return 42

    return Test()


def func2():
    class Test(trampoline_module.test_override_cache_helper):
        pass

    return Test()
