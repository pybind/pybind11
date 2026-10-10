from __future__ import annotations

import os
import sys

import test_cmake_build

assert isinstance(__file__, str)  # Test this is properly set

assert test_cmake_build.add(1, 2) == 3

expect_abi3 = os.environ.get("PYBIND11_EXPECT_ABI3")
if expect_abi3 and sys.platform != "win32":
    assert ".abi3" in expect_abi3, f"expected an abi3 module name, got {expect_abi3}"
print(f"{sys.argv[1]} imports, runs, and adds: 1 + 2 = 3")
