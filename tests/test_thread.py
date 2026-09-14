from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading

import pytest

import env
import pybind11_tests
from pybind11_tests import thread as m


class Thread(threading.Thread):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.e = None

    def run(self):
        try:
            for i in range(10):
                self.fn(i, i)
        except Exception as e:
            self.e = e

    def join(self):
        super().join()
        if self.e:
            raise self.e


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test_implicit_conversion():
    a = Thread(m.test)
    b = Thread(m.test)
    c = Thread(m.test)
    for x in [a, b, c]:
        x.start()
    for x in [c, b, a]:
        x.join()


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test_implicit_conversion_no_gil():
    a = Thread(m.test_no_gil)
    b = Thread(m.test_no_gil)
    c = Thread(m.test_no_gil)
    for x in [a, b, c]:
        x.start()
    for x in [c, b, a]:
        x.join()


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test_bind_shared_instance():
    nb_threads = 4
    b = threading.Barrier(nb_threads)

    def access_shared_instance():
        b.wait()
        for _ in range(1000):
            m.EmptyStruct.SharedInstance  # noqa: B018

    threads = [
        threading.Thread(target=access_shared_instance) for _ in range(nb_threads)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
@pytest.mark.skipif(not m.defined_PYBIND11_HAS_STD_BARRIER, reason="no <barrier>")
@pytest.mark.skipif(env.sys_is_gil_enabled(), reason="Deadlock with the GIL")
def test_pythread_state_clear_destructor():
    class Foo:
        def __del__(self):
            m.acquire_gil()

    m.test_pythread_state_clear_destructor(Foo)


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
@pytest.mark.skipif(env.ANDROID or env.IOS, reason="Requires subprocess support")
@pytest.mark.skipif(
    not env.PY_GIL_DISABLED, reason="The internals lock is a no-op with the GIL"
)
def test_dispatch_does_not_need_internals_lock():
    """A bound call must not block on the internals mutex while another thread holds it.

    Regression test for #6159 (item 59): cpp_function::dispatcher() used to reach its
    function_record through get_function_record_PyTypeObject(), which takes the internals lock
    on every call. Runs in a subprocess because on a regression the call under test blocks
    until the holder's watchdog fires.
    """
    script = textwrap.dedent(
        f"""
        import sys

        sys.path.insert(0, {os.path.dirname(pybind11_tests.__file__)!r})

        from pybind11_tests import thread as m

        m.dispatch_noop()  # Warm up any one-time initialization.

        # Positive control: the holder holds the very mutex that internals users take, so a
        # call that needs it can only return after the holder's watchdog fires.
        m.start_internals_lock_holder(1.0)
        m.take_internals_lock()
        assert m.release_internals_lock_holder(), "take_internals_lock() did not block"

        # The call under test: a plain bound call must not need the internals lock at all.
        m.start_internals_lock_holder(5.0)
        m.dispatch_noop()
        released_by_watchdog = m.release_internals_lock_holder()
        assert not released_by_watchdog, "cpp_function::dispatcher() needed the internals lock"
        """
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as ex:
        pytest.fail(
            f"Subprocess did not finish within {ex.timeout} s (deadlock?).\n"
            f"Output:\n{ex.stdout}\n{ex.stderr}"
        )
    assert proc.returncode == 0, (
        f"Subprocess failed with exit code {proc.returncode}.\n"
        f"Output:\n{proc.stdout}\n{proc.stderr}"
    )
