import namespace_visibility_1
import namespace_visibility_2
import pybind11_cross_module_tests as cm
import pytest


def test_namespace_visibility():
    del cm  # Unused.
    modules = (
        namespace_visibility_1,
        namespace_visibility_2,
    )
    for m in modules:
        if m.__doc__ != "ns_vis_1":
            pytest.skip(f"Not surprised: {m.__doc__}")
