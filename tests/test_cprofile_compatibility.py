from __future__ import annotations

import cProfile as profile
import io
import pstats

import cprofile_compatibility_ext


def _get_pstats(profiler):
    output = io.StringIO()
    stats = pstats.Stats(profiler, stream=output)
    stats.sort_stats("cumtime")
    stats.print_stats()
    return output.getvalue()


class PyClass:
    def py_member_func(self) -> None:
        return cprofile_compatibility_ext.CppClass().member_func_return_secret()


def test_free_func_return_secret():
    profiler = profile.Profile()
    profiler.enable()
    assert cprofile_compatibility_ext.free_func_return_secret() == 102
    profiler.disable()
    stats_output = _get_pstats(profiler)
    assert "LETS_SEE_WHAT_WE_GET_FREE_FUNC" in stats_output


def test_member_func_return_secret():
    obj = PyClass()
    profiler = profile.Profile()
    profiler.enable()
    assert obj.py_member_func() == 203
    profiler.disable()
    stats_output = _get_pstats(profiler)
    assert "LETS_SEE_WHAT_WE_GET_MEMBER_FUNC" in stats_output
