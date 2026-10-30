from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import test_cmake_build

assert isinstance(__file__, str)  # Test this is properly set

assert test_cmake_build.add(1, 2) == 3

expect_abi3 = os.environ.get("PYBIND11_EXPECT_ABI3")
if expect_abi3 and sys.platform != "win32":
    assert ".abi3" in expect_abi3, f"expected an abi3 module name, got {expect_abi3}"
print(f"{sys.argv[1]} imports, runs, and adds: 1 + 2 = 3")


if test_cmake_build.limited_api:
    # A fresh, small module leaves MethodType uncached until the first bound-method read.
    script = textwrap.dedent(
        """
        import builtins
        import threading
        import time

        import test_cmake_build as m

        entered = threading.Event()
        second_started = threading.Event()
        original_import = builtins.__import__

        def slow_import(name, *args, **kwargs):
            if name == "types":
                entered.set()
                assert second_started.wait(10)
                time.sleep(0.2)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = slow_import
        instance = m.ColdMethodType()
        results = []

        def read_method():
            method = getattr(instance, "value")
            results.append(method())

        def second_read():
            second_started.set()
            read_method()

        first = threading.Thread(target=read_method)
        second = threading.Thread(target=second_read)
        first.start()
        assert entered.wait(10), "MethodType was already cached"
        second.start()
        first.join()
        second.join()
        assert results == [7, 7]
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
