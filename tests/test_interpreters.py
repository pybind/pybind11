from __future__ import annotations

import os
import pickle
import sys

import pytest


@pytest.mark.skipif(
    sys.platform.startswith("emscripten") or sys.version_info < (3, 12),
    reason="Requires independent subinterpreter support",
)
def test_interpreters():
    """Makes sure the internals object differs across subinterpreters"""

    sys.path.append(".")

    i = None
    try:
        # 3.14+
        import interpreters as i
    except ImportError:
        try:
            # 3.13
            import _interpreters as i
        except ImportError:
            try:
                # 3.12
                import _xxsubinterpreters as i
            except ImportError:
                pytest.skip("Test requires a the interpreters stdlib module")
                return

    import mod_test_interpreters as m

    code = """
import mod_test_interpreters as m
import pickle
with open(pipeo, 'wb') as f:
    pickle.dump(m.internals_at(), f)
"""

    interp1 = i.create()
    interp2 = i.create()
    try:
        pipei, pipeo = os.pipe()
        i.run_string(interp1, code, shared={"pipeo": pipeo})
        with open(pipei, "rb") as f:
            res1 = pickle.load(f)

        pipei, pipeo = os.pipe()
        i.run_string(interp2, code, shared={"pipeo": pipeo})
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
        i.destroy(interp1)
        i.destroy(interp2)

    assert res1 != m.internals_at(), "internals should differ from main interpreter"
    assert res2 != m.internals_at(), "internals should differ from main interpreter"
    assert res1 != res2, "internals should differ between interpreters"
