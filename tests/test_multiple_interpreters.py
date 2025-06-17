from __future__ import annotations

import contextlib
import os
import pickle
import sys

import pytest

CONCURRENT_INTERPRETERS_SUPPORT = sys.version_info >= (3, 14) and (
    sys.version_info != (3, 14, 0, "beta", 1)
    or sys.version_info != (3, 14, 0, "beta", 2)
)


def get_interpreters(*, modern: bool):
    if modern and CONCURRENT_INTERPRETERS_SUPPORT:
        from concurrent import interpreters

        def create():
            return contextlib.closing(interpreters.create())

        def run_string(
            interp: interpreters.Interpreter,
            code: str,
            *,
            shared: dict[str, object] | None = None,
        ) -> Exception | None:
            if shared:
                interp.prepare_main(**shared)
            try:
                interp.exec(code)
                return None
            except interpreters.ExecutionFailed as err:
                return err

        return run_string, create

    if sys.version_info >= (3, 12):
        interpreters = pytest.importorskip(
            "_interpreters" if sys.version_info >= (3, 13) else "_xxsubinterpreters"
        )

        @contextlib.contextmanager
        def create(config: str = ""):
            try:
                if config:
                    interp = interpreters.create(config)
                else:
                    interp = interpreters.create()
            except TypeError:
                pytest.skip(f"interpreters module needs to support {config} config")

            try:
                yield interp
            finally:
                interpreters.destroy(interp)

        def run_string(
            interp: int, code: str, shared: dict[str, object] | None = None
        ) -> Exception | None:
            kwargs = {"shared": shared} if shared else {}
            return interpreters.run_string(interp, code, **kwargs)

        return run_string, create

    pytest.skip("Test requires the interpreters stdlib module")


@pytest.mark.skipif(
    sys.platform.startswith("emscripten"), reason="Requires loadable modules"
)
def test_independent_subinterpreters():
    """Makes sure the internals object differs across independent subinterpreters"""

    sys.path.append(".")

    run_string, create = get_interpreters(modern=True)

    m = pytest.importorskip("mod_per_interpreter_gil")

    if not m.defined_PYBIND11_HAS_SUBINTERPRETER_SUPPORT:
        pytest.skip("Does not have subinterpreter support compiled in")

    code = """
import mod_per_interpreter_gil as m
import pickle
with open(pipeo, 'wb') as f:
    pickle.dump(m.internals_at(), f)
"""

    with create() as interp1, create() as interp2:
        try:
            res0 = run_string(interp1, "import mod_shared_interpreter_gil")
            if res0 is not None:
                res0 = str(res0)
        except Exception as e:
            res0 = str(e)

        pipei, pipeo = os.pipe()
        run_string(interp1, code, shared={"pipeo": pipeo})
        with open(pipei, "rb") as f:
            res1 = pickle.load(f)

        pipei, pipeo = os.pipe()
        run_string(interp2, code, shared={"pipeo": pipeo})
        with open(pipei, "rb") as f:
            res2 = pickle.load(f)

        # do this while the two interpreters are active
        import mod_per_interpreter_gil as m2

        assert m.internals_at() == m2.internals_at(), (
            "internals should be the same within the main interpreter"
        )

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

    run_string, create = get_interpreters(modern=False)

    m = pytest.importorskip("mod_shared_interpreter_gil")

    if not m.defined_PYBIND11_HAS_SUBINTERPRETER_SUPPORT:
        pytest.skip("Does not have subinterpreter support compiled in")

    code = """
import mod_shared_interpreter_gil as m
import pickle
with open(pipeo, 'wb') as f:
    pickle.dump(m.internals_at(), f)
"""

    with create("legacy") as interp1:
        pipei, pipeo = os.pipe()
        run_string(interp1, code, shared={"pipeo": pipeo})
        with open(pipei, "rb") as f:
            res1 = pickle.load(f)

        # do this while the other interpreter is active
        import mod_shared_interpreter_gil as m2

        assert m.internals_at() == m2.internals_at(), (
            "internals should be the same within the main interpreter"
        )

    assert res1 != m.internals_at(), "internals should differ from main interpreter"

    # do this after the other interpreters are destroyed and only one remains
    import mod_shared_interpreter_gil as m3

    assert m.internals_at() == m3.internals_at(), (
        "internals should be the same within the main interpreter"
    )
