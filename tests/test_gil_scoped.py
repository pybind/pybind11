import multiprocessing
import threading

import pytest

import env
from pybind11_tests import gil_scoped as m


class ExtendedVirtClass(m.VirtClass):
    def virtual_func(self):
        pass

    def pure_virtual_func(self):
        pass


def test_callback_py_obj():
    m.test_callback_py_obj(lambda: None)


def test_callback_std_func():
    m.test_callback_std_func(lambda: None)


def test_callback_virtual_func():
    extended = ExtendedVirtClass()
    m.test_callback_virtual_func(extended)


def test_callback_pure_virtual_func():
    extended = ExtendedVirtClass()
    m.test_callback_pure_virtual_func(extended)


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


# Intentionally putting human review in the loop here, to guard against accidents.
VARS_BEFORE_ALL_BASIC_TESTS = dict(vars())  # Make a copy of the dict (critical).
ALL_BASIC_TESTS = (
    test_callback_py_obj,
    test_callback_std_func,
    test_callback_virtual_func,
    test_callback_pure_virtual_func,
    test_cross_module_gil_released,
    test_cross_module_gil_acquired,
    test_cross_module_gil_inner_custom_released,
    test_cross_module_gil_inner_custom_acquired,
    test_cross_module_gil_inner_pybind11_released,
    test_cross_module_gil_inner_pybind11_acquired,
    test_cross_module_gil_nested_custom_released,
    test_cross_module_gil_nested_custom_acquired,
    test_cross_module_gil_nested_pybind11_released,
    test_cross_module_gil_nested_pybind11_acquired,
    test_release_acquire,
    test_nested_acquire,
    test_multi_acquire_release_cross_module,
)


def test_all_basic_tests_completeness():
    num_found = 0
    for key, value in VARS_BEFORE_ALL_BASIC_TESTS.items():
        if not key.startswith("test_"):
            continue
        assert value in ALL_BASIC_TESTS
        num_found += 1
    assert len(ALL_BASIC_TESTS) == num_found


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


def _run_in_threads(target, num_threads, parallel):
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        if parallel:
            threads.append(thread)
        else:
            thread.join()
    for thread in threads:
        thread.join()


# m.defined_THREAD_SANITIZER is used below to skip tests triggering this error (#2754):
# ThreadSanitizer: starting new threads after multi-threaded fork is not supported.

# TODO: FIXME, sometimes returns -11 (segfault) instead of 0 on macOS Python 3.9
@pytest.mark.skipif(
    m.defined_THREAD_SANITIZER, reason="Not compatible with ThreadSanitizer"
)
@pytest.mark.parametrize("test_fn", ALL_BASIC_TESTS)
def test_run_in_process_one_thread(test_fn):
    """Makes sure there is no GIL deadlock when running in a thread.

    It runs in a separate process to be able to stop and assert if it deadlocks.
    """
    assert _run_in_process(_run_in_threads, test_fn, num_threads=1, parallel=False) == 0


# TODO: FIXME on macOS Python 3.9
@pytest.mark.skipif(
    m.defined_THREAD_SANITIZER, reason="Not compatible with ThreadSanitizer"
)
@pytest.mark.parametrize("test_fn", ALL_BASIC_TESTS)
def test_run_in_process_multiple_threads_parallel(test_fn):
    """Makes sure there is no GIL deadlock when running in a thread multiple times in parallel.

    It runs in a separate process to be able to stop and assert if it deadlocks.
    """
    exitcode = _run_in_process(_run_in_threads, test_fn, num_threads=8, parallel=True)
    if exitcode is None and env.PYPY and env.WIN:  # Seems to be flaky.
        pytest.skip("Ignoring unexpected exitcode None (PYPY WIN)")
    assert exitcode == 0


# TODO: FIXME on macOS Python 3.9
@pytest.mark.skipif(
    m.defined_THREAD_SANITIZER, reason="Not compatible with ThreadSanitizer"
)
@pytest.mark.parametrize("test_fn", ALL_BASIC_TESTS)
def test_run_in_process_multiple_threads_sequential(test_fn):
    """Makes sure there is no GIL deadlock when running in a thread multiple times sequentially.

    It runs in a separate process to be able to stop and assert if it deadlocks.
    """
    assert _run_in_process(_run_in_threads, test_fn, num_threads=8, parallel=False) == 0


# TODO: FIXME on macOS Python 3.9
@pytest.mark.parametrize("test_fn", ALL_BASIC_TESTS)
def test_run_in_process_direct(test_fn):
    """Makes sure there is no GIL deadlock when using processes.

    This test is for completion, but it was never an issue.
    """
    if m.defined_THREAD_SANITIZER and test_fn in (
        test_cross_module_gil_nested_custom_released,
        test_cross_module_gil_nested_custom_acquired,
        test_cross_module_gil_nested_pybind11_released,
        test_cross_module_gil_nested_pybind11_acquired,
        test_multi_acquire_release_cross_module,
    ):
        pytest.skip("Not compatible with ThreadSanitizer")
    exitcode = _run_in_process(test_fn)
    if exitcode is None and env.PYPY and env.WIN:  # Seems to be flaky.
        pytest.skip("Ignoring unexpected exitcode None (PYPY WIN)")
    assert exitcode == 0
