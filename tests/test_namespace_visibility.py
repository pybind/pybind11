# This is not really a unit test, but probing environment/toolchain/platform-specific
# behavior under the exact same conditions as normal tests.
# The results are useful to understanding the effects of, e.g., removing
# `-fvisibility=hidden` or `__attribute__((visibility("hidden")))`, or linking
# extensions statically with the core Python interpreter.

import itertools

import namespace_visibility_1
import namespace_visibility_2
import pytest


def test_namespace_visibility():
    modules = (
        namespace_visibility_1,
        namespace_visibility_1,  # .submodule,
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
    if all_unique_pointers_observed != "AAC:AAc:AaC:Aac:aAC:aAc:aaC:aac":
        pytest.skip(
            f"UNUSUAL all_unique_pointers_observed: {all_unique_pointers_observed}"
        )
