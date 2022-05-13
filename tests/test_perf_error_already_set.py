import time

import pytest

from pybind11_tests import perf_error_already_set as m


def find_call_repetitions(
    callable,
    time_delta_floor=1.0e-6,
    target_elapsed_secs_multiplier=1.05,  # Empirical.
    target_elapsed_secs_tolerance=0.05,
    max_iterations=100,
    call_repetitions_first_pass=1000,
    call_repetitions_target_elapsed_secs=1.0,
):
    td_target = call_repetitions_target_elapsed_secs * target_elapsed_secs_multiplier
    crd = call_repetitions_first_pass
    for _ in range(max_iterations):
        t0 = time.time()
        callable(crd)
        td = time.time() - t0
        crd = max(1, int(td_target * crd / max(td, time_delta_floor)))
        if abs(td - td_target) / td_target < target_elapsed_secs_tolerance:
            return crd
    raise RuntimeError("find_call_repetitions failure: max_iterations exceeded.")


def pr1895_original_foo(num_iterations):
    assert num_iterations >= 0
    while num_iterations:
        m.pr1895_original_foo()
        num_iterations -= 1


@pytest.mark.parametrize(
    "perf_name",
    [
        "pure_unwind",
        "err_set_unwind_err_clear",
        "err_set_error_already_set",
        "error_already_set_restore",
        "pr1895_original_foo",
    ],
)
def test_perf(perf_name):
    print(flush=True)
    if perf_name == "pr1895_original_foo":
        callable = pr1895_original_foo
    else:
        callable = getattr(m, perf_name)
    call_repetitions = find_call_repetitions(callable)
    t0 = time.time()
    callable(call_repetitions)
    secs = time.time() - t0
    u_secs = secs * 10e6
    per_call = u_secs / call_repetitions
    print(f"PERF {perf_name},{call_repetitions},{secs:.3f},{per_call:.6f}", flush=True)
