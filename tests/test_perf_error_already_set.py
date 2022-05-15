import time

import pytest

from pybind11_tests import perf_error_already_set as m


def recurse_first_then_call(depth, callable, call_repetitions, call_error_string):
    if depth:
        recurse_first_then_call(
            depth - 1, callable, call_repetitions, call_error_string
        )
    else:
        if call_error_string is None:
            callable(call_repetitions)
        else:
            callable(call_repetitions, call_error_string)


def find_call_repetitions(
    recursion_depth,
    callable,
    call_error_string,
    time_delta_floor=1.0e-6,
    target_elapsed_secs_multiplier=1.05,  # Empirical.
    target_elapsed_secs_tolerance=0.05,
    max_iterations=100,
    call_repetitions_first_pass=1000,
    call_repetitions_target_elapsed_secs=1.0,  # 1.0 for real benchmarking.
):
    td_target = call_repetitions_target_elapsed_secs * target_elapsed_secs_multiplier
    crd = call_repetitions_first_pass
    for _ in range(max_iterations):
        t0 = time.time()
        recurse_first_then_call(recursion_depth, callable, crd, call_error_string)
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
        "err_set_err_clear",
        "err_set_error_already_set",
        "err_set_err_fetch",
        "error_already_set_restore",
        "err_fetch_err_restore",
        "pr1895_original_foo",
    ],
)
def test_perf(perf_name):
    print(flush=True)
    if perf_name == "pr1895_original_foo":
        callable = pr1895_original_foo
    else:
        callable = getattr(m, perf_name)
    if perf_name in ("pure_unwind", "pr1895_original_foo"):
        call_error_string_args = [None]
    else:
        call_error_string_args = [False, True]
    first_per_call_results = None
    for ix_outer, call_error_string in enumerate(call_error_string_args):
        per_call_results = []
        for recursion_depth in (0, 100):
            call_repetitions = find_call_repetitions(
                recursion_depth, callable, call_error_string
            )
            t0 = time.time()
            recurse_first_then_call(
                recursion_depth, callable, call_repetitions, call_error_string
            )
            secs = time.time() - t0
            u_secs = secs * 10e6
            per_call = u_secs / call_repetitions
            if first_per_call_results is None:
                relative_to_first = ""
            else:
                rel_percent = first_per_call_results[ix_outer] / per_call * 100
                relative_to_first = f",{rel_percent:.2f}%"
            print(
                f"PERF {perf_name},{recursion_depth},{call_repetitions},{call_error_string},"
                f"{secs:.3f}s,{per_call:.6f}μs{relative_to_first}",
                flush=True,
            )
            per_call_results.append(per_call)
        if first_per_call_results is None:
            first_per_call_results = tuple(per_call_results)
