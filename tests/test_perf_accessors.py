import time

from pybind11_tests import perf_accessors as m


def find_num_iterations(
    callable,
    callable_args,
    num_iterations_first_pass,
    num_iterations_target_elapsed_secs,
    time_delta_floor=1.0e-6,
    target_elapsed_secs_multiplier=1.05,  # Empirical.
    target_elapsed_secs_tolerance=0.05,
    search_max_iterations=100,
):
    td_target = num_iterations_target_elapsed_secs * target_elapsed_secs_multiplier
    num_iterations = num_iterations_first_pass
    for _ in range(search_max_iterations):
        t0 = time.time()
        callable(num_iterations, *callable_args)
        td = time.time() - t0
        num_iterations = max(
            1, int(td_target * num_iterations / max(td, time_delta_floor))
        )
        if abs(td - td_target) / td_target < target_elapsed_secs_tolerance:
            return num_iterations
    raise RuntimeError("find_num_iterations failure: search_max_iterations exceeded.")


def test_perf_list_accessor():
    print(flush=True)
    inc_refs = m.perf_list_accessor(0, 0)
    print(f"inc_refs={inc_refs}", flush=True)
    if inc_refs is not None:
        assert inc_refs == [1, 1]
    num_iterations = find_num_iterations(
        callable=m.perf_list_accessor,
        callable_args=(0,),
        num_iterations_first_pass=100000,
        num_iterations_target_elapsed_secs=2.0,
    )
    for test_id in range(2):
        t0 = time.time()
        m.perf_list_accessor(num_iterations, test_id)
        secs = time.time() - t0
        u_secs = secs * 10e6
        per_call = u_secs / num_iterations
        print(
            f"PERF {test_id},{num_iterations},{secs:.3f}s,{per_call:.6f}μs",
            flush=True,
        )
