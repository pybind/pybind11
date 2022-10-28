import multiprocessing
import threading

import pytest

import env
from pybind11_tests import gil_scoped as m


def _run_in_process(target, *args, **kwargs):
    """Runs target in process and returns its exitcode after 10s (None if still alive)."""
    process = multiprocessing.Process(target=target, args=args, kwargs=kwargs)
    process.daemon = True
    try:
        process.start()
        # Do not need to wait much, 10s should be more than enough.
        process.join(timeout=10)
        return process.exitcode
    finally:
        if process.is_alive():
            process.terminate()


def _python_to_cpp_to_python():
    """Calls different C++ functions that come back to Python."""

    class ExtendedVirtClass(m.VirtClass):
        def virtual_func(self):
            pass

        def pure_virtual_func(self):
            pass

    extended = ExtendedVirtClass()
    m.test_callback_py_obj(lambda: None)
    m.test_callback_std_func(lambda: None)
    m.test_callback_virtual_func(extended)
    m.test_callback_pure_virtual_func(extended)

    m.test_cross_module_gil_released()
    m.test_cross_module_gil_acquired()
    m.test_cross_module_gil_inner_custom_released()
    m.test_cross_module_gil_inner_custom_acquired()
    m.test_cross_module_gil_inner_pybind11_released()
    m.test_cross_module_gil_inner_pybind11_acquired()
    m.test_cross_module_gil_nested_custom_released()
    m.test_cross_module_gil_nested_custom_acquired()
    m.test_cross_module_gil_nested_pybind11_released()
    m.test_cross_module_gil_nested_pybind11_acquired()

    assert m.test_release_acquire(0xAB) == "171"
    assert m.test_nested_acquire(0xAB) == "171"

    for bits in range(16 * 8):
        internals_ids = m.test_multi_acquire_release_cross_module(bits)
        assert len(internals_ids) == 2 if bits % 8 else 1


def _python_to_cpp_to_python_from_threads(num_threads, parallel=False):
    """Calls different C++ functions that come back to Python, from Python threads."""
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=_python_to_cpp_to_python)
        thread.daemon = True
        thread.start()
        if parallel:
            threads.append(thread)
        else:
            thread.join()
    for thread in threads:
        thread.join()


# TODO: FIXME, sometimes returns -11 (segfault) instead of 0 on macOS Python 3.9
def test_python_to_cpp_to_python_from_thread():
    """Makes sure there is no GIL deadlock when running in a thread.

    It runs in a separate process to be able to stop and assert if it deadlocks.
    """
    assert _run_in_process(_python_to_cpp_to_python_from_threads, 1) == 0


# TODO: FIXME on macOS Python 3.9
def test_python_to_cpp_to_python_from_thread_multiple_parallel():
    """Makes sure there is no GIL deadlock when running in a thread multiple times in parallel.

    It runs in a separate process to be able to stop and assert if it deadlocks.
    """
    exitcode = _run_in_process(_python_to_cpp_to_python_from_threads, 8, parallel=True)
    if exitcode is None and env.PYPY and env.WIN:  # Seems to be flaky.
        pytest.skip("Ignoring unexpected exitcode None (PYPY WIN)")
    assert exitcode == 0


# TODO: FIXME on macOS Python 3.9
def test_python_to_cpp_to_python_from_thread_multiple_sequential():
    """Makes sure there is no GIL deadlock when running in a thread multiple times sequentially.

    It runs in a separate process to be able to stop and assert if it deadlocks.
    """
    assert (
        _run_in_process(_python_to_cpp_to_python_from_threads, 8, parallel=False) == 0
    )


# TODO: FIXME on macOS Python 3.9
def test_python_to_cpp_to_python_from_process():
    """Makes sure there is no GIL deadlock when using processes.

    This test is for completion, but it was never an issue.
    """
    exitcode = _run_in_process(_python_to_cpp_to_python)
    if exitcode is None and env.PYPY and env.WIN:  # Seems to be flaky.
        pytest.skip("Ignoring unexpected exitcode None (PYPY WIN)")
    assert exitcode == 0


def test_cross_module_gil_released():
    """Makes sure that the GIL can be acquired by another module from a GIL-released state."""
    m.test_cross_module_gil_released()  # Should not raise a SIGSEGV


def test_cross_module_gil_acquired():
    """Makes sure that the GIL can be acquired by another module from a GIL-acquired state."""
    m.test_cross_module_gil_acquired()  # Should not raise a SIGSEGV


def test_cross_module_gil_inner_custom_released():
    """Makes sure that the GIL can be acquired/released by another module
    from a GIL-released state using custom locking logic."""
    m.test_cross_module_gil_inner_custom_released()


def test_cross_module_gil_inner_custom_acquired():
    """Makes sure that the GIL can be acquired/acquired by another module
    from a GIL-acquired state using custom locking logic."""
    m.test_cross_module_gil_inner_custom_acquired()


def test_cross_module_gil_inner_pybind11_released():
    """Makes sure that the GIL can be acquired/released by another module
    from a GIL-released state using pybind11 locking logic."""
    m.test_cross_module_gil_inner_pybind11_released()


def test_cross_module_gil_inner_pybind11_acquired():
    """Makes sure that the GIL can be acquired/acquired by another module
    from a GIL-acquired state using pybind11 locking logic."""
    m.test_cross_module_gil_inner_pybind11_acquired()


def test_cross_module_gil_nested_custom_released():
    """Makes sure that the GIL can be nested acquired/released by another module
    from a GIL-released state using custom locking logic."""
    m.test_cross_module_gil_nested_custom_released()


def test_cross_module_gil_nested_custom_acquired():
    """Makes sure that the GIL can be nested acquired/acquired by another module
    from a GIL-acquired state using custom locking logic."""
    m.test_cross_module_gil_nested_custom_acquired()


def test_cross_module_gil_nested_pybind11_released():
    """Makes sure that the GIL can be nested acquired/released by another module
    from a GIL-released state using pybind11 locking logic."""
    m.test_cross_module_gil_nested_pybind11_released()


def test_cross_module_gil_nested_pybind11_acquired():
    """Makes sure that the GIL can be nested acquired/acquired by another module
    from a GIL-acquired state using pybind11 locking logic."""
    m.test_cross_module_gil_nested_pybind11_acquired()


def test_release_acquire():
    assert m.test_release_acquire(0xAB) == "171"


def test_nested_acquire():
    assert m.test_nested_acquire(0xAB) == "171"


def test_multi_acquire_release_cross_module():
    for bits in range(16 * 8):
        internals_ids = m.test_multi_acquire_release_cross_module(bits)
        assert len(internals_ids) == 2 if bits % 8 else 1
