from __future__ import annotations

import gc
import multiprocessing
import weakref

import pytest

import env  # noqa: F401
import pybind11_tests
from pybind11_tests import invalid_holder_access as m

XFAIL_REASON = "Known issues: https://github.com/pybind/pybind11/pull/5654"


try:
    spawn_context = multiprocessing.get_context("spawn")
except ValueError:
    spawn_context = None


@pytest.mark.skipif(
    pybind11_tests.cpp_std_num < 14,
    reason="std::{unique_ptr,make_unique} not available in C++11",
)
def test_create_vector():
    vec = m.create_vector()
    assert vec.size() == 4
    assert not vec.is_empty()
    assert vec[0] is None
    assert vec[1] == 1
    assert vec[2] == "test"
    assert vec[3] == ()


def test_no_init():
    with pytest.raises(TypeError, match=r"No constructor defined"):
        m.VectorOwns4PythonObjects()
    vec = m.VectorOwns4PythonObjects.__new__(m.VectorOwns4PythonObjects)
    with pytest.raises(TypeError, match=r"No constructor defined"):
        vec.__init__()


def manual_new_target():
    # Repeatedly trigger allocation without initialization (raw malloc'ed) to
    # detect uninitialized memory bugs.
    for _ in range(32):
        # The holder is a pointer variable while the C++ ctor is not called.
        vec = m.VectorOwns4PythonObjects.__new__(m.VectorOwns4PythonObjects)
        if vec.is_empty():  # <= this line could cause a segfault
            # The C++ compiler initializes container correctly.
            assert vec.size() == 0
        else:
            raise SystemError(
                "Segmentation Fault: The C++ compiler initializes container incorrectly."
            )
        vec.append(1)
        vec.append(2)
        vec.append(3)
        vec.append(4)


# This test might succeed on some platforms with some compilers, but it is not
# guaranteed to work everywhere. It is marked as non-strict xfail.
@pytest.mark.xfail(reason=XFAIL_REASON, raises=SystemError, strict=False)
@pytest.mark.skipif(spawn_context is None, reason="spawn context not available")
def test_manual_new():
    process = spawn_context.Process(
        target=manual_new_target,
        name="manual_new_target",
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise SystemError(
            "Segmentation Fault: The C++ compiler initializes container incorrectly."
        )


@pytest.mark.skipif("env.PYPY or env.GRAALPY", reason="Cannot reliably trigger GC")
@pytest.mark.skipif(
    pybind11_tests.cpp_std_num < 14,
    reason="std::{unique_ptr,make_unique} not available in C++11",
)
def test_gc_traverse():
    vec = m.create_vector()
    vec[3] = (vec, vec)

    wr = weakref.ref(vec)
    assert wr() is vec
    del vec
    for _ in range(10):
        gc.collect()
    assert wr() is None
