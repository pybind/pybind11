from __future__ import annotations

import gc
import multiprocessing
import pickle
import signal
import sys
import weakref

import pytest

import env  # noqa: F401
import pybind11_tests
from pybind11_tests import invalid_holder_access as m

XFAIL_REASON = "Known issues: https://github.com/pybind/pybind11/pull/5654"


try:
    import multiprocessing

    spawn_context = multiprocessing.get_context("spawn")
except (ImportError, ValueError):
    spawn_context = None


def check_segfault(target):
    """Check if the target function causes a segmentation fault.

    The check should be done in a separate process to avoid crashing the main
    process with the segfault.
    """
    process = spawn_context.Process(target=target)
    process.start()
    process.join()
    rc = abs(process.exitcode)
    if 128 < rc < 256:
        rc -= 128
    assert rc in (
        0,
        signal.SIGSEGV,
        signal.SIGABRT,
        0xC0000005,  # STATUS_ACCESS_VIOLATION on Windows
    )
    if rc != 0:
        raise SystemError(
            "Segmentation Fault: The C++ compiler initializes container incorrectly."
        )


def test_no_init():
    with pytest.raises(TypeError, match=r"No constructor defined"):
        m.VecOwnsObjs()
    vec = m.VecOwnsObjs.__new__(m.VecOwnsObjs)
    with pytest.raises(TypeError, match=r"No constructor defined"):
        vec.__init__()


def manual_new_target():
    # Repeatedly trigger allocation without initialization (raw malloc'ed) to
    # detect uninitialized memory bugs.
    for _ in range(32):
        # The holder is a pointer variable while the C++ ctor is not called.
        vec = m.VecOwnsObjs.__new__(m.VecOwnsObjs)
        if vec.is_empty():  # <= this line could cause a segfault
            # The C++ compiler initializes container correctly.
            assert len(vec) == 0
        else:
            # The program is not supposed to reach here. It will abort with
            # SIGSEGV on `vec.is_empty()`.
            sys.exit(signal.SIGSEGV)  # manually trigger SIGSEGV if not raised


# This test might succeed on some platforms with some compilers, but it is not
# guaranteed to work everywhere. It is marked as non-strict xfail.
@pytest.mark.xfail(reason=XFAIL_REASON, raises=SystemError, strict=False)
@pytest.mark.skipif(spawn_context is None, reason="spawn context not available")
@pytest.mark.skipif(
    sys.platform.startswith("emscripten"),
    reason="Requires multiprocessing",
)
def test_manual_new():
    """
    Manually calling the constructor (__new__) of a class that has C++ STL
    container will allocate memory without initializing it can cause a
    segmentation fault.
    """
    check_segfault(manual_new_target)


@pytest.mark.skipif(
    pybind11_tests.cpp_std_num < 14,
    reason="std::{unique_ptr,make_unique} not available in C++11",
)
def test_set_state_with_error_no_segfault_if_gc_checks_holder_has_initialized():
    """
    The ``tp_traverse`` and ``tp_clear`` functions should check if the holder
    has been initialized before traversing or clearing the C++ STL container.
    """
    m.VecOwnsObjs.set_should_check_holder_initialization(True)

    vec = m.create_vector((1, 2, 3, 4))

    m.VecOwnsObjs.set_should_raise_error_on_set_state(False)
    pickle.loads(pickle.dumps(vec))

    # During the unpickling process, Python firstly allocates the object with
    # the `__new__` method and then calls the `__setstate__`` method to set the
    # state of the object.
    #
    #     obj = cls.__new__(cls)
    #     obj.__setstate__(state)
    #
    # The `__init__` method is not called during unpickling.
    # So, if the `__setstate__` method raises an error, the object will be in
    # an uninitialized state.
    m.VecOwnsObjs.set_should_raise_error_on_set_state(True)
    serialized = pickle.dumps(vec)
    with pytest.raises(
        RuntimeError,
        match="raise error on set_state for testing",
    ):
        pickle.loads(serialized)


def unpicklable_with_error_target():
    m.VecOwnsObjs.set_should_check_holder_initialization(False)
    m.VecOwnsObjs.set_should_raise_error_on_set_state(True)

    vec = m.create_vector((1, 2, 3, 4))
    serialized = pickle.dumps(vec)

    # Repeatedly trigger allocation without initialization (raw malloc'ed) to
    # detect uninitialized memory bugs.
    for _ in range(32):
        # During the unpickling process, Python firstly allocates the object with
        # the `__new__` method and then calls the `__setstate__`` method to set the
        # state of the object.
        #
        #     obj = cls.__new__(cls)
        #     obj.__setstate__(state)
        #
        # The `__init__` method is not called during unpickling.
        # So, if the `__setstate__` method raises an error, the object will be in
        # an uninitialized state. The GC will traverse the uninitialized C++ STL
        # container and cause a segmentation fault.
        try:  # noqa: SIM105
            pickle.loads(serialized)
        except RuntimeError:
            pass


# This test might succeed on some platforms with some compilers, but it is not
# guaranteed to work everywhere. It is marked as non-strict xfail.
@pytest.mark.xfail(reason=XFAIL_REASON, raises=SystemError, strict=False)
@pytest.mark.skipif(spawn_context is None, reason="spawn context not available")
@pytest.mark.skipif(
    pybind11_tests.cpp_std_num < 14,
    reason="std::{unique_ptr,make_unique} not available in C++11",
)
def test_set_state_with_error_will_segfault_if_gc_does_not_check_holder_has_initialized():
    m.VecOwnsObjs.set_should_check_holder_initialization(False)

    vec = m.create_vector((1, 2, 3, 4))

    m.VecOwnsObjs.set_should_raise_error_on_set_state(False)
    pickle.loads(pickle.dumps(vec))

    # See comments above.
    check_segfault(unpicklable_with_error_target)


@pytest.mark.skipif("env.PYPY or env.GRAALPY", reason="Cannot reliably trigger GC")
@pytest.mark.skipif(
    pybind11_tests.cpp_std_num < 14,
    reason="std::{unique_ptr,make_unique} not available in C++11",
)
def test_gc():
    vec = m.create_vector(())
    vec.append((vec, vec))

    wr = weakref.ref(vec)
    assert wr() is vec
    del vec
    for _ in range(10):
        gc.collect()
    assert wr() is None
