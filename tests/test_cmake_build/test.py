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


if getattr(test_cmake_build, "limited_api", False):
    # Each fresh, small module leaves the selected Python type cache uninitialized.
    script = textwrap.dedent(
        """
        import builtins
        import sys
        import threading
        import time

        import test_cmake_build as m

        import_name = sys.argv[1]
        entered = threading.Event()
        second_started = threading.Event()
        original_import = builtins.__import__

        def slow_import(name, *args, **kwargs):
            if name == import_name:
                entered.set()
                assert second_started.wait(10)
                time.sleep(0.2)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = slow_import
        results = []
        errors = []
        if import_name == "types":
            instance = m.ColdMethodType()
            def call():
                return getattr(instance, "value")()
        else:
            def call():
                return m.chrono_roundtrip(1.0)

        def read_method():
            try:
                results.append(call())
            except BaseException as exc:
                errors.append(exc)

        def second_read():
            second_started.set()
            read_method()

        first = threading.Thread(target=read_method)
        second = threading.Thread(target=second_read)
        first.start()
        assert entered.wait(10), f"{import_name} was already cached"
        second.start()
        first.join()
        second.join()
        assert not errors, errors
        assert len(results) == 2
        if import_name == "types":
            assert results == [7, 7]
        else:
            assert all(result.total_seconds() == 1.0 for result in results)
        """
    )
    for import_name in ("types", "datetime"):
        subprocess.run(
            [sys.executable, "-c", script, import_name], check=True, timeout=30
        )
