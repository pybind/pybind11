from __future__ import annotations

import contextlib
import os
import pickle
import sys
import textwrap

import pytest

import env
import pybind11_tests

if env.IOS:
    pytest.skip("Subinterpreters not supported on iOS", allow_module_level=True)

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

    sys.path.insert(0, os.path.dirname(pybind11_tests.__file__))

    run_string, create = get_interpreters(modern=True)

    import mod_per_interpreter_gil as m

    if not m.defined_PYBIND11_HAS_SUBINTERPRETER_SUPPORT:
        pytest.skip("Does not have subinterpreter support compiled in")

    code = textwrap.dedent(
        """
        import mod_per_interpreter_gil as m
        import pickle
        with open(pipeo, 'wb') as f:
            pickle.dump(m.internals_at(), f)
        """
    ).strip()

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

    sys.path.insert(0, os.path.dirname(pybind11_tests.__file__))

    import mod_per_interpreter_gil as m

    if not m.defined_PYBIND11_HAS_SUBINTERPRETER_SUPPORT:
        pytest.skip("Does not have subinterpreter support compiled in")

    from concurrent import interpreters

    code = textwrap.dedent(
        """
        import mod_per_interpreter_gil as m

        values.put_nowait(m.internals_at())
        """
    ).strip()

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

    sys.path.insert(0, os.path.dirname(pybind11_tests.__file__))

    run_string, create = get_interpreters(modern=False)

    import mod_shared_interpreter_gil as m

    if not m.defined_PYBIND11_HAS_SUBINTERPRETER_SUPPORT:
        pytest.skip("Does not have subinterpreter support compiled in")

    code = textwrap.dedent(
        """
        import mod_shared_interpreter_gil as m
        import pickle
        with open(pipeo, 'wb') as f:
            pickle.dump(m.internals_at(), f)
        """
    ).strip()

    with create("legacy") as interp1:
        pipei, pipeo = os.pipe()
        run_string(interp1, code, shared={"pipeo": pipeo})
        with open(pipei, "rb") as f:
            res1 = pickle.load(f)

    assert res1 != m.internals_at(), "internals should differ from main interpreter"


PREAMBLE_CODE = textwrap.dedent(
    f"""
    def test():
        import sys

        sys.path.insert(0, {os.path.dirname(env.__file__)!r})
        sys.path.insert(0, {os.path.dirname(pybind11_tests.__file__)!r})

        import collections
        import mod_per_interpreter_gil_with_singleton as m

        objects = m.get_objects_in_singleton()
        expected = [
            type(None),               # static type: shared between interpreters
            tuple,                    # static type: shared between interpreters
            list,                     # static type: shared between interpreters
            dict,                     # static type: shared between interpreters
            collections.OrderedDict,  # static type: shared between interpreters
            collections.defaultdict,  # heap type: dynamically created per interpreter
            collections.deque,        # heap type: dynamically created per interpreter
        ]
        # Check that we have the expected objects. Avoid IndexError by checking lengths first.
        assert len(objects) == len(expected), (
            f"Expected {{expected!r}} ({{len(expected)}}), got {{objects!r}} ({{len(objects)}})."
        )
        # The first ones are static types shared between interpreters.
        assert objects[:-2] == expected[:-2], (
            f"Expected static objects {{expected[:-2]!r}}, got {{objects[:-2]!r}}."
        )
        # The last two are heap types created per-interpreter.
        # The expected objects are dynamically imported from `collections`.
        assert objects[-2:] == expected[-2:], (
            f"Expected heap objects {{expected[-2:]!r}}, got {{objects[-2:]!r}}."
        )

        assert hasattr(m, 'MyClass'), "Module missing MyClass"
        assert hasattr(m, 'MyGlobalError'), "Module missing MyGlobalError"
        assert hasattr(m, 'MyLocalError'), "Module missing MyLocalError"
        assert hasattr(m, 'MyEnum'), "Module missing MyEnum"
    """
).lstrip()


@pytest.mark.skipif(
    sys.platform.startswith("emscripten"), reason="Requires loadable modules"
)
@pytest.mark.skipif(not CONCURRENT_INTERPRETERS_SUPPORT, reason="Requires 3.14.0b3+")
def test_import_module_with_singleton_per_interpreter():
    """Tests that a singleton storing Python objects works correctly per-interpreter"""
    from concurrent import interpreters

    code = f"{PREAMBLE_CODE.strip()}\n\ntest()\n"
    with contextlib.closing(interpreters.create()) as interp:
        interp.exec(code)


@pytest.mark.skipif(
    sys.platform.startswith("emscripten"), reason="Requires loadable modules"
)
@pytest.mark.skipif(not CONCURRENT_INTERPRETERS_SUPPORT, reason="Requires 3.14.0b3+")
def test_import_in_subinterpreter_after_main():
    """Tests that importing a module in a subinterpreter after the main interpreter works correctly"""
    env.check_script_success_in_subprocess(
        PREAMBLE_CODE
        + textwrap.dedent(
            """
            import contextlib
            import gc
            from concurrent import interpreters

            test()

            interp = None
            with contextlib.closing(interpreters.create()) as interp:
                interp.call(test)

            del interp
            for _ in range(5):
                gc.collect()
            """
        )
    )

    env.check_script_success_in_subprocess(
        PREAMBLE_CODE
        + textwrap.dedent(
            """
            import contextlib
            import gc
            import random
            from concurrent import interpreters

            test()

            interps = interp = None
            with contextlib.ExitStack() as stack:
                interps = [
                    stack.enter_context(contextlib.closing(interpreters.create()))
                    for _ in range(8)
                ]
                random.shuffle(interps)
                for interp in interps:
                    interp.call(test)

            del interps, interp, stack
            for _ in range(5):
                gc.collect()
            """
        )
    )


@pytest.mark.skipif(
    sys.platform.startswith("emscripten"), reason="Requires loadable modules"
)
@pytest.mark.skipif(not CONCURRENT_INTERPRETERS_SUPPORT, reason="Requires 3.14.0b3+")
def test_import_in_subinterpreter_before_main():
    """Tests that importing a module in a subinterpreter before the main interpreter works correctly"""
    env.check_script_success_in_subprocess(
        PREAMBLE_CODE
        + textwrap.dedent(
            """
            import contextlib
            import gc
            from concurrent import interpreters

            interp = None
            with contextlib.closing(interpreters.create()) as interp:
                interp.call(test)

            test()

            del interp
            for _ in range(5):
                gc.collect()
            """
        )
    )

    env.check_script_success_in_subprocess(
        PREAMBLE_CODE
        + textwrap.dedent(
            """
            import contextlib
            import gc
            from concurrent import interpreters

            interps = interp = None
            with contextlib.ExitStack() as stack:
                interps = [
                    stack.enter_context(contextlib.closing(interpreters.create()))
                    for _ in range(8)
                ]
                for interp in interps:
                    interp.call(test)

            test()

            del interps, interp, stack
            for _ in range(5):
                gc.collect()
            """
        )
    )

    env.check_script_success_in_subprocess(
        PREAMBLE_CODE
        + textwrap.dedent(
            """
            import contextlib
            import gc
            from concurrent import interpreters

            interps = interp = None
            with contextlib.ExitStack() as stack:
                interps = [
                    stack.enter_context(contextlib.closing(interpreters.create()))
                    for _ in range(8)
                ]
                for interp in interps:
                    interp.call(test)

                test()

            del interps, interp, stack
            for _ in range(5):
                gc.collect()
            """
        )
    )


@pytest.mark.skipif(
    sys.platform.startswith("emscripten"), reason="Requires loadable modules"
)
@pytest.mark.skipif(not CONCURRENT_INTERPRETERS_SUPPORT, reason="Requires 3.14.0b3+")
def test_import_in_subinterpreter_concurrently():
    """Tests that importing a module in multiple subinterpreters concurrently works correctly"""
    env.check_script_success_in_subprocess(
        PREAMBLE_CODE
        + textwrap.dedent(
            """
            import gc
            from concurrent.futures import InterpreterPoolExecutor, as_completed

            futures = future = None
            with InterpreterPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(test) for _ in range(32)]
                for future in as_completed(futures):
                    future.result()
            del futures, future, executor

            for _ in range(5):
                gc.collect()
            """
        )
    )
