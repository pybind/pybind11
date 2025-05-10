from __future__ import annotations

import os
import pickle
import sys

import pytest


@pytest.mark.skipif(
    sys.platform.startswith("emscripten"), reason="Requires loadable modules"
)
def test_interpreters():
    """Makes sure the internals object differs across subinterpreters"""

    sys.path.append(".")

    if sys.version_info >= (3, 14):
        import interpreters
    elif sys.version_info >= (3, 13):
        import _interpreters as interpreters
    elif sys.version_info >= (3, 12):
        import _xxsubinterpreters as interpreters
    else:
        pytest.skip("Test requires a the interpreters stdlib module")

    import mod_test_interpreters as m

    code = """
import mod_test_interpreters as m
import pickle
with open(pipeo, 'wb') as f:
    pickle.dump(m.internals_at(), f)
"""

    interp1 = interpreters.create()
    interp2 = interpreters.create()
    try:
        pipei, pipeo = os.pipe()
        interpreters.run_string(interp1, code, shared={"pipeo": pipeo})
        with open(pipei, "rb") as f:
            res1 = pickle.load(f)

        pipei, pipeo = os.pipe()
        interpreters.run_string(interp2, code, shared={"pipeo": pipeo})
        with open(pipei, "rb") as f:
            res2 = pickle.load(f)

        # do this while the two interpreters are active
        import mod_test_interpreters as m2

        print(dir(m))
        print(dir(m2))
        assert m.internals_at() == m2.internals_at(), (
            "internals should be the same within the main interpreter"
        )
    finally:
        interpreters.destroy(interp1)
        interpreters.destroy(interp2)

    assert res1 != m.internals_at(), "internals should differ from main interpreter"
    assert res2 != m.internals_at(), "internals should differ from main interpreter"
    assert res1 != res2, "internals should differ between interpreters"
