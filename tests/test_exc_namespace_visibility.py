# This is not really a unit test, but probing environment/toolchain/platform-specific
# behavior under the exact same conditions as normal tests.
# The results are useful to understanding the effects of, e.g., removing
# `-fvisibility=hidden` or `__attribute__((visibility("hidden")))`, or linking
# extensions statically with the core Python interpreter.

# NOTE
# ====
# The "exc_" in "test_exc_namespace_visibility.py" is a workaround, to avoid a
# test_cross_module_exception_translator (test_exceptions.py) failure. This
# test has to be imported (by pytest) before test_exceptions.py; pytest sorts
# lexically. See https://github.com/pybind/pybind11/pull/4054 for more information.

import itertools

import namespace_visibility_1
import namespace_visibility_2
import pytest

# Please take a quick look at namespace_visibility.h first, to see what is being probed.
#
# EXPECTED is for -fvisibility=hidden or equivalent, as recommended in the docs.
EXPECTED_ALL_UNIQUE_POINTERS_OBSERVED = "AAC:AAc:AaC:Aac:aAC:aAc:aaC:aac"
#                                        ^^^
#          namespace_visibility_1 pointer|||
# namespace_visibility_1.submodule pointer|| (identical letters means same pointer)
#            namespace_visibility_2 pointer|
# Upper-case: namespace visibility unspecified
# Lower-case: namespace visibility hidden
# This test probes all 2**3 combinations of u/h ** number-of-sub/modules.
#
# Also observed:
# AAA:AAc:AaC:Aac:aAC:aAc:aaC:aac  -fvisibility=default Linux
# AAA:AAc:AaA:Aac:aAC:aAc:aaC:aac  -fvisibility=default macOS
# AAA:AAa:AaA:Aaa:aAA:aAa:aaA:aaa  everything linked statically


def test_namespace_visibility():
    modules = (
        namespace_visibility_1,
        namespace_visibility_1.submodule,
        namespace_visibility_2,
    )
    unique_pointer_labels = "ABC"
    unique_pointers_observed = []
    # u = visibility unspecified
    # h = visibility hidden
    for visibility in itertools.product(*([("u", "h")] * len(modules))):
        # See functions in namespace_visibility_*.cpp
        func = "ns_vis_" + "".join(visibility) + "_func"
        ptrs = []
        uq_ptrs_obs = ""
        for vis, m in zip(visibility, modules):
            ptr = getattr(m, func)()
            ptrs.append(ptr)
            lbl = unique_pointer_labels[ptrs.index(ptr)]
            if vis == "h":
                # Encode u/h info as upper/lower case to make the final result
                # as compact as possible.
                lbl = lbl.lower()
            uq_ptrs_obs += lbl
        unique_pointers_observed.append(uq_ptrs_obs)
    all_unique_pointers_observed = ":".join(unique_pointers_observed)
    if all_unique_pointers_observed != EXPECTED_ALL_UNIQUE_POINTERS_OBSERVED:
        pytest.skip(
            f"UNUSUAL all_unique_pointers_observed: {all_unique_pointers_observed}"
        )
