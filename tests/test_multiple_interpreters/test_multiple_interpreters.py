from __future__ import annotations

import contextlib
import os
import pickle
import sys

import pytest

# 3.14.0b3+, though sys.implementation.supports_isolated_interpreters is being added in b4
# Can be simplified when we drop support for the first three betas
CONCURRENT_INTERPRETERS_SUPPORT = (
    sys.version_info >= (3, 14)
    and (
        sys.version_info != (3, 14, 0, "beta", 1)
        and sys.version_info != (3, 14, 0, "beta", 2)
    )
    and (
        sys.version_info == (3, 14, 0, "beta", 3)
        or sys.implementation.supports_isolated_interpreters
    )
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

    assert "does not support loading in subinterpreters" in res0, (
        "cannot use shared_gil in a default subinterpreter"
    )
    assert res1 != m.internals_at(), "internals should differ from main interpreter"
    assert res2 != m.internals_at(), "internals should differ from main interpreter"
    assert res1 != res2, "internals should differ between interpreters"


@pytest.mark.skipif(
    sys.platform.startswith("emscripten"), reason="Requires loadable modules"
)
@pytest.mark.skipif(not CONCURRENT_INTERPRETERS_SUPPORT, reason="Requires 3.14.0b3+")
def test_independent_subinterpreters_modern():
    """Makes sure the internals object differs across independent subinterpreters. Modern (3.14+) syntax."""

    sys.path.append(".")

    m = pytest.importorskip("mod_per_interpreter_gil")

    if not m.defined_PYBIND11_HAS_SUBINTERPRETER_SUPPORT:
        pytest.skip("Does not have subinterpreter support compiled in")

    from concurrent import interpreters

    code = """
import mod_per_interpreter_gil as m

values.put_nowait(m.internals_at())
"""

    with contextlib.closing(interpreters.create()) as interp1, contextlib.closing(
        interpreters.create()
    ) as interp2:
        with pytest.raises(
            interpreters.ExecutionFailed,
            match="does not support loading in subinterpreters",
        ):
            interp1.exec("import mod_shared_interpreter_gil")

        values = interpreters.create_queue()
        interp1.prepare_main(values=values)
        interp1.exec(code)
        res1 = values.get_nowait()

        interp2.prepare_main(values=values)
        interp2.exec(code)
        res2 = values.get_nowait()

    assert res1 != m.internals_at(), "internals should differ from main interpreter"
    assert res2 != m.internals_at(), "internals should differ from main interpreter"
    assert res1 != res2, "internals should differ between interpreters"


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

    assert res1 != m.internals_at(), "internals should differ from main interpreter"
