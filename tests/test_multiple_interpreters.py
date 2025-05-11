from __future__ import annotations

import os
import pickle
import sys

import pytest


@pytest.mark.skipif(
    sys.platform.startswith("emscripten"), reason="Requires loadable modules"
)
def test_independent_subinterpreters():
    """Makes sure the internals object differs across independent subinterpreters"""

    sys.path.append(".")

    if sys.version_info >= (3, 14):
        import interpreters
    elif sys.version_info >= (3, 13):
        import _interpreters as interpreters
    elif sys.version_info >= (3, 12):
        import _xxsubinterpreters as interpreters
    else:
        pytest.skip("Test requires a the interpreters stdlib module")

    import mod_per_interpreter_gil as m

    code = """
import mod_per_interpreter_gil as m
import pickle
with open(pipeo, 'wb') as f:
    pickle.dump(m.internals_at(), f)
"""

    interp1 = interpreters.create()
    interp2 = interpreters.create()
    try:
        try:
            res0 = interpreters.run_string(interp1, "import mod_shared_interpreter_gil")
            if res0 is not None:
                res0 = res0.msg
        except Exception as e:
            res0 = str(e)

        pipei, pipeo = os.pipe()
        interpreters.run_string(interp1, code, shared={"pipeo": pipeo})
        with open(pipei, "rb") as f:
            res1 = pickle.load(f)

        pipei, pipeo = os.pipe()
        interpreters.run_string(interp2, code, shared={"pipeo": pipeo})
        with open(pipei, "rb") as f:
            res2 = pickle.load(f)

        # do this while the two interpreters are active
        import mod_per_interpreter_gil as m2

        assert m.internals_at() == m2.internals_at(), (
            "internals should be the same within the main interpreter"
        )
    finally:
        interpreters.destroy(interp1)
        interpreters.destroy(interp2)

    assert "does not support loading in subinterpreters" in res0, (
        "cannot use shared_gil in a default subinterpreter"
    )
    assert res1 != m.internals_at(), "internals should differ from main interpreter"
    assert res2 != m.internals_at(), "internals should differ from main interpreter"
    assert res1 != res2, "internals should differ between interpreters"

    # do this after the two interpreters are destroyed and only one remains
    import mod_per_interpreter_gil as m3

    assert m.internals_at() == m3.internals_at(), (
        "internals should be the same within the main interpreter"
    )


@pytest.mark.skipif(
    sys.platform.startswith("emscripten"), reason="Requires loadable modules"
)
def test_dependent_subinterpreters():
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

    import mod_shared_interpreter_gil as m

    code = """
import mod_shared_interpreter_gil as m
import pickle
with open(pipeo, 'wb') as f:
    pickle.dump(m.internals_at(), f)
"""

    try:
        interp1 = interpreters.create("legacy")
    except TypeError:
        pytest.skip("interpreters module needs to support legacy config")

    try:
        pipei, pipeo = os.pipe()
        interpreters.run_string(interp1, code, shared={"pipeo": pipeo})
        with open(pipei, "rb") as f:
            res1 = pickle.load(f)

        # do this while the other interpreter is active
        import mod_shared_interpreter_gil as m2

        assert m.internals_at() == m2.internals_at(), (
            "internals should be the same within the main interpreter"
        )
    finally:
        interpreters.destroy(interp1)

    assert res1 != m.internals_at(), "internals should differ from main interpreter"

    # do this after the other interpreters are destroyed and only one remains
    import mod_shared_interpreter_gil as m3

    assert m.internals_at() == m3.internals_at(), (
        "internals should be the same within the main interpreter"
    )
