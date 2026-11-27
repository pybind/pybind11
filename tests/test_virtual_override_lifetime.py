from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

import env
import pybind11_tests
from pybind11_tests import virtual_override_lifetime as m


def _owner_removed_during_lookup():
    # A dynamic attribute provider can release the Python owner during lookup.
    owners = []

    class Observer(m.Observer):
        def __getattribute__(self, name):
            if name == "on_event":
                owners.clear()
                raise AttributeError(name)
            return super().__getattribute__(name)

        def __getattr__(self, name):
            if name == "on_event":
                assert observer_ref() is not None
                return lambda: 2
            raise AttributeError(name)

    owners.append(Observer())
    observer_ref = weakref.ref(owners[0])
    dispatcher = m.Dispatcher(owners[0])
    assert dispatcher.notify() == 2
    gc.collect()
    assert observer_ref() is None


def _observer_replaced_during_dispatch():
    started = Event()

    class Observer(m.Observer):
        def on_event(self):
            started.set()
            return 2

    observer = Observer()
    observer_ref = weakref.ref(observer)
    dispatcher = m.Dispatcher(observer)
    assert dispatcher.notify() == 2
    del observer
    gc.collect()
    # This is the documented shared_ptr-holder behavior: C++ remains alive and
    # virtual calls fall back to the C++ implementation after the wrapper dies.
    assert observer_ref() is None
    assert dispatcher.notify() == 1

    observer = Observer()
    dispatcher.set_observer(observer)
    started.clear()
    with ThreadPoolExecutor(max_workers=1) as pool:
        calls = pool.submit(dispatcher.run)
        try:
            assert started.wait(10), "the worker did not start dispatching"
            del observer
            for _ in range(20000):
                dispatcher.set_observer(Observer())
        finally:
            dispatcher.stop()
        assert calls.result() > 0


@pytest.mark.skipif(
    not env.CPYTHON, reason="Requires CPython reference-counting behavior"
)
@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
@pytest.mark.skipif(env.ANDROID or env.IOS, reason="Requires subprocess support")
@pytest.mark.parametrize(
    "case", [_owner_removed_during_lookup, _observer_replaced_during_dispatch]
)
def test_virtual_override_lifetime(case):
    # Isolate a possible native failure; collection can also enable the GIL on a
    # free-threaded interpreter, so explicitly disable it in that child process.
    paths = [os.path.dirname(pybind11_tests.__file__), os.path.dirname(__file__)]
    script = textwrap.dedent(
        f"""
        import sys
        sys.path[:0] = {paths!r}
        from test_virtual_override_lifetime import {case.__name__}
        {case.__name__}()
        """
    )
    command = [sys.executable, "-X", "dev"]
    if env.PY_GIL_DISABLED:
        command += ["-X", "gil=0"]
    result = subprocess.run(
        [*command, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
