# Copyright (c) 2025 Hudson River Trading LLC <opensource@hudson-trading.com>
from __future__ import annotations

import collections
import gc
import itertools
import sys
import sysconfig
import threading
import time
import weakref

import pytest

import env

# t1, t2, t3 all use the default foreign_interop::full(), meaning they
# auto-import and auto-export all bindings. They each use a different
# PYBIND11_INTERNALS_VERSION so they are foreign to each other.
#   t1: internals v100, shared_ptr holder
#   t2: internals v200, smart_holder
#   t3: internals v300, smart_holder + raw C API framework (RawShared)
# t4 uses foreign_interop::on_request() -- no auto import/export.
# t5 uses foreign_interop::disabled() -- foreign interop fully disabled.
#
# Upon import, bindings are defined for the functions, but not for the
# types (Shared and SharedEnum) until you call bind_types().
# t3's RawShared is created via create_raw_binding().

import test_foreign_1 as t1
import test_foreign_2 as t2
import test_foreign_3 as t3
import test_foreign_4 as t4
import test_foreign_5 as t5

free_threaded = hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()
types_are_immortal = sys.implementation.name in ("graalpy", "pypy") or (
    sysconfig.get_config_var("Py_GIL_DISABLED") and sys.version_info < (3, 14)
)


def delattr_and_ensure_destroyed(*specs):
    wrs = []
    for mod, name in specs:
        wrs.append(weakref.ref(getattr(mod, name)))
        delattr(mod, name)

    for _ in range(10):
        gc.collect()
        if all(wr() is None for wr in wrs):
            break
    else:
        pytest.fail(
            f"Could not delete bindings such as "
            f"{next(wr for wr in wrs if wr() is not None)!r}"
        )


@pytest.fixture(autouse=True)
def clean_after():
    yield
    if sys.implementation.name in ("pypy", "graalpy"):
        pytest.gc_collect()
    if sys.implementation.name != "graalpy":
        t3.clear_foreign_bindings()

    # Try to remove types so each test starts fresh. On CPython, pybind11's
    # internal type maps (registered_types_cpp, native_enum_type_map) keep
    # strong references to the type objects, so they might not actually be
    # destroyed. In that case, bind_types() will notice the existing types
    # and re-export them if needed rather than trying to re-create them.
    for mod in (t1, t2, t3, t4, t5):
        for name in ("Shared", "SharedEnum", "RawShared"):
            if hasattr(mod, name):
                try:
                    delattr(mod, name)
                except AttributeError:
                    pass
    for _ in range(5):
        gc.collect()

    for mod in (t1, t2, t3, t4, t5):
        mod.pull_stats()


def check_stats(mod, **entries):
    if mod is None or sys.implementation.name == "graalpy":
        return
    if sys.implementation.name == "pypy":
        pytest.gc_collect()
    stats = mod.pull_stats()
    if stats["move"] == entries.get("move", 0) + 1:
        # Allow an extra move+destroy pair to account for older compilers
        # not doing RVO like we expect.
        entries["move"] = entries.get("move", 0) + 1
        entries["destroy"] = entries.get("destroy", 0) + 1
    for name, value in entries.items():
        assert stats.pop(name) == value
    assert all(val == 0 for val in stats.values())


global_counter = itertools.count()


def expect(from_mod, to_mod, pattern, **extra):
    """Test creating objects in from_mod and consuming them in to_mod.

    pattern is one of:
      "local"    - full interop (native binding), shared_ptr use_count == 2
      "foreign"  - foreign interop works, shared_ptr use_count == 1 (new control block)
      "isolated" - types exist but can't be passed across (TypeError)
      "none"     - can't even create the object (no binding)
    """
    outcomes = {}
    extra_info = {}
    owner_mod = None

    for idx, suffix in enumerate(("", "_sp", "_up", "_enum")):
        create = getattr(from_mod, f"make{suffix}")
        check = getattr(to_mod, f"check{suffix}")
        thing = suffix.lstrip("_") or "value"
        value = idx * 1000 + next(global_counter)
        if thing == "enum":
            value = (value % 2) + 1
        try:
            obj = create(value)
        except Exception as ex:
            outcomes[thing] = None
            extra_info[thing] = ex
            continue
        if owner_mod is None:
            owner_mod = sys.modules[type(obj).__module__]
        try:
            roundtripped = check(obj)
        except Exception as ex:
            outcomes[thing] = False
            extra_info[thing] = ex
            continue
        assert roundtripped == value, "instance appears corrupted"
        if thing == "sp":
            outcomes[thing] = to_mod.uses(obj)
        else:
            outcomes[thing] = True

    expected = {}
    if pattern == "local":
        # unique_ptr works locally only for smart_holder modules (t2, t3, t4, t5)
        expected = {
            "value": True,
            "sp": 2,
            "up": to_mod not in (t1,),
            "enum": True,
        }
    elif pattern == "foreign":
        expected = {"value": True, "sp": 1, "up": False, "enum": True}
    elif pattern == "isolated":
        expected = {"value": False, "sp": False, "up": False, "enum": False}
    elif pattern == "none":
        expected = {"value": None, "sp": None, "up": None, "enum": None}
    else:
        pytest.fail("unknown pattern")
    expected.update(extra)
    assert outcomes == expected, f"extra_info={extra_info}"

    obj = None

    # When returning by value, we have a construction in from_mod,
    # move to owner_mod, destruction in from_mod (after make() returns)
    # and destruction in owner mod (when the pyobject dies).
    #
    # When returning shared_ptr, the construction and destruction both
    # occur in from_mod since shared_ptr's deleter is set at creation time.
    #
    # When returning unique_ptr, the construction occurs in from_mod and
    # destruction (when the pyobject dies) occurs in owner_mod.
    expect_stats = {mod: collections.Counter() for mod in (from_mod, to_mod, owner_mod)}
    expect_stats[from_mod].update(
        ["construct", "destroy", "construct", "destroy", "construct"]
    )
    # value move+destroy
    expect_stats[owner_mod].update(["move", "destroy"])
    # unique_ptr destroy
    if owner_mod is None and from_mod is t1:
        pass
    else:
        expect_stats[owner_mod or from_mod].update(["destroy"])
    for mod, stats in expect_stats.items():
        check_stats(mod, **stats)


# =====================================================================
# Test 1: Automatic interoperability between full-mode modules
# =====================================================================

def test_auto_interop_full():
    """t1 and t2 both use foreign_interop::full(). Once they bind their types,
    the types should be auto-exported and auto-imported, allowing seamless
    cross-module usage without any manual import/export calls."""
    t1.bind_types()
    t2.bind_types()

    # Each module can use its own types locally
    expect(t1, t1, "local")
    expect(t2, t2, "local")

    # Cross-module: t1's types work in t2 and vice versa
    expect(t1, t2, "foreign")
    expect(t2, t1, "foreign")

    # Verify type ownership: each module returns its own type
    assert type(t1.make(1)) is t1.Shared
    assert type(t2.make(2)) is t2.Shared


# =====================================================================
# Test 2: Exception translator sharing
# =====================================================================

@pytest.mark.skipif(
    (env.MACOS and env.PYPY) or env.ANDROID,
    reason="same issue as test_exceptions.py test_cross_module_exception_translator",
)
def test_auto_interop_exceptions():
    """t2 registers an exception translator for SharedExc. With full()
    mode, this translator is automatically shared, so t1 (which has no
    translator) can use it."""
    t1.bind_types()
    t2.bind_types()

    # t2 has its own translator
    with pytest.raises(ValueError, match="Shared.200"):
        t2.throw_shared(200)

    # t1 uses t2's translator via foreign exception translation
    with pytest.raises(ValueError, match="Shared.100"):
        t1.throw_shared(100)


# =====================================================================
# Test 3: C API framework interop (non-C++ types need explicit import)
# =====================================================================

def test_c_api_requires_explicit_import():
    """t3's RawShared is bound via the C API, not C++. Auto-import doesn't
    work for non-C++ types, so we need explicit import_foreign<Shared>()."""
    t1.bind_types()
    t3.create_raw_binding()

    # t1->t3 works because t1.Shared is auto-exported (C++) and t3
    # auto-imports C++ types. t1's SharedEnum is also auto-exported.
    expect(t1, t3, "foreign")

    # t3->t1: RawShared is not C++ so it's not auto-imported.
    # However, t3.make_enum uses t1's SharedEnum (auto-imported),
    # so enums DO work in the t3->t1 direction.
    expect(t3, t1, "isolated", enum=True)

    # After explicit import, t3->t1 works fully
    t1.import_foreign_explicit(t3.RawShared)
    expect(t3, t1, "foreign")


# =====================================================================
# Test 4: on_request mode
# =====================================================================

def test_on_request_mode():
    """t4 uses foreign_interop::on_request(). It should not auto-import
    or auto-export. But explicit import/export should work."""
    t1.bind_types()
    t4.bind_types()

    # Each module can use its own types locally
    expect(t1, t1, "local")
    expect(t4, t4, "local")

    # t4 didn't export, so t1 can't use t4's types (even though t1
    # auto-imports). t1 exported, but t4 doesn't auto-import.
    expect(t1, t4, "isolated")
    expect(t4, t1, "isolated")

    # Explicitly export t4's type and import t1's type in t4
    t4.export_to_foreign(t4.Shared)
    t4.export_to_foreign(t4.SharedEnum)
    t4.import_foreign(t1.Shared)
    t4.import_foreign(t1.SharedEnum)

    # Now t4 can accept t1's types
    expect(t1, t4, "foreign")

    # And t1 can accept t4's types (t1 auto-imports, t4 just exported)
    expect(t4, t1, "foreign")


# =====================================================================
# Test 5: disabled mode
# =====================================================================

def test_disabled_mode():
    """t5 uses foreign_interop::disabled(). import_foreign and
    export_to_foreign should raise exceptions."""
    t1.bind_types()
    t5.bind_types()

    # t5 can use its own types
    expect(t5, t5, "local")

    # Cross-module doesn't work
    expect(t1, t5, "isolated")
    expect(t5, t1, "isolated")

    # import_foreign and export_to_foreign both raise
    with pytest.raises(RuntimeError, match="foreign_interop::disabled"):
        t5.import_foreign(t1.Shared)
    with pytest.raises(RuntimeError, match="foreign_interop::disabled"):
        t5.export_to_foreign(t5.Shared)


# =====================================================================
# Test 6: Import/export error handling
# =====================================================================

def test_import_export_errors():
    """Test various error conditions for import_foreign and export_to_foreign."""
    t1.bind_types()
    t2.bind_types()
    t3.create_raw_binding()

    # Can't import a type that doesn't have __pymetabind_binding__
    with pytest.raises(
        RuntimeError, match="type does not define a __pymetabind_binding__"
    ):
        t2.import_foreign(t1.Convertible)

    # Can't export a type that's not from our pybind11 domain
    with pytest.raises(RuntimeError, match="not a pybind11 class or enum"):
        t3.export_to_foreign(t2.Shared)

    with pytest.raises(RuntimeError, match="not a pybind11 class or enum"):
        t3.export_to_foreign(t2.SharedEnum)

    # Exporting should be idempotent
    t2.export_to_foreign(t2.Shared)
    t2.export_to_foreign(t2.Shared)
    t2.export_to_foreign(t2.SharedEnum)
    t2.export_to_foreign(t2.SharedEnum)

    # Can't import our own type
    with pytest.raises(RuntimeError, match="type is not foreign"):
        t2.import_foreign(t2.Shared)

    # Can't import a non-C++ type without explicit template arg
    with pytest.raises(RuntimeError, match=r"is not written in C\+\+"):
        t2.import_foreign(t3.RawShared)

    # Explicit import works and is idempotent
    t2.import_foreign_explicit(t3.RawShared)
    t2.import_foreign_explicit(t3.RawShared)

    # Can't import same type as different C++ type
    with pytest.raises(
        RuntimeError, match=r"was already imported as a different C\+\+ type"
    ):
        t2.import_foreign_wrong_type(t3.RawShared)


# =====================================================================
# Test 7: Manual import priority
# =====================================================================

def test_manual_import_priority():
    """When a type has multiple foreign bindings, import_foreign() should
    move the imported one to the front (preferred for to-Python conversions).

    t3 has its own RawShared binding (from the C API framework), so its make()
    always returns RawShared (the local binding is preferred). Instead, we test
    priority using t4 (on_request mode) which has no auto-imported bindings."""
    t1.bind_types()
    t2.bind_types()

    # Import t1's binding into t4 first, then t2's
    t4.import_foreign(t1.Shared)
    t4.import_foreign(t1.SharedEnum)

    # t4 should prefer t1.Shared since we imported it first (it's at the front)
    obj = t4.make(42)
    assert type(obj) is t1.Shared

    # Now import t2.Shared -- it should move to the front
    t4.import_foreign(t2.Shared)
    t4.import_foreign(t2.SharedEnum)
    obj2 = t4.make(43)
    assert type(obj2) is t2.Shared

    # Re-import t1.Shared -- it should move back to the front
    t4.import_foreign(t1.Shared)
    t4.import_foreign(t1.SharedEnum)
    obj3 = t4.make(44)
    assert type(obj3) is t1.Shared


# =====================================================================
# Test 8: shared_ptr use counts
# =====================================================================

def test_shared_ptr_use_count():
    """Foreign shared_ptr creates a new control block (use_count=1),
    while local reuses the existing one (use_count=2)."""
    t1.bind_types()
    t2.bind_types()

    sp1 = t1.make_sp(10)
    sp2 = t2.make_sp(20)

    # Local: shared_ptr is shared, use_count includes both the C++ sp
    # and the Python reference
    assert t1.uses(sp1) == 2
    assert t2.uses(sp2) == 2

    # Foreign: new control block, use_count == 1
    assert t2.uses(sp1) == 1
    assert t1.uses(sp2) == 1


# =====================================================================
# Test 9: unique_ptr transfer rejected across foreign boundary
# =====================================================================

def test_unique_ptr_foreign_rejected():
    """Cannot pass unique_ptr across foreign boundary because ownership
    can't be transferred to a foreign framework."""
    t1.bind_types()
    t2.bind_types()

    # unique_ptr works locally for smart_holder (t2)
    obj2_local = t2.make(20)
    assert t2.check_up(obj2_local) == 20

    # unique_ptr does NOT work across foreign boundary
    obj1_for_t2 = t1.make(10)
    with pytest.raises((TypeError, RuntimeError)):
        t2.check_up(obj1_for_t2)  # t1's Shared passed to t2's check_up


# =====================================================================
# Test 10: Implicit conversion from foreign types
# =====================================================================

def test_implicit_conversion_from_foreign():
    """py::implicitly_convertible<Shared, Convertible>() in t1 should
    work when source is a foreign-bound Shared from t2 or t3."""
    t1.bind_types()
    t2.bind_types()
    t3.create_raw_binding()
    t1.import_foreign_explicit(t3.RawShared)

    s1 = t1.make(10)
    s2 = t2.make(11)
    s3r = t3.make(12)

    # All can be implicitly converted to Convertible
    assert t1.test_implicit(s1) == 10
    assert t1.test_implicit(s2) == 11
    assert t1.test_implicit(s3r) == 12


# =====================================================================
# Test 11: Remove binding
# =====================================================================

@pytest.mark.skipif(types_are_immortal, reason="can't GC type object on this platform")
def test_remove_binding():
    """Removing __pymetabind_binding__ should make
    it unavailable for foreign interop."""
    t1.bind_types()
    t2.bind_types()

    # Cross-module works
    expect(t1, t2, "foreign")
    expect(t2, t1, "foreign")

    # Remove t2's binding capsule
    del t2.Shared.__pymetabind_binding__
    del t2.SharedEnum.__pymetabind_binding__
    pytest.gc_collect()

    # t1 can no longer use t2's types
    expect(t2, t1, "isolated")

    # But t2 can still use t1's types (t1 still has its binding)
    expect(t1, t2, "foreign")

    # Re-export t2's types
    t2.export_to_foreign(t2.Shared)
    t2.export_to_foreign(t2.SharedEnum)

    # Works again
    expect(t2, t1, "foreign")


# =====================================================================
# Test 12: Remove and recreate raw binding
# =====================================================================

@pytest.mark.skipif(types_are_immortal, reason="can't GC type object on this platform")
def test_remove_raw_binding():
    """Removing and recreating t3.RawShared should be handled gracefully."""
    t3.create_raw_binding()
    t2.bind_types()
    t2.import_foreign_explicit(t3.RawShared)

    expect(t3, t2, "foreign", enum=True)

    # Remove the binding
    delattr_and_ensure_destroyed((t3, "RawShared"))

    # Recreate it
    t3.create_raw_binding()

    # Need to re-import since it's a new binding
    t2.import_foreign_explicit(t3.RawShared)
    expect(t3, t2, "foreign", enum=True)


# =====================================================================
# Test 13: Concurrent access
# =====================================================================

@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test_concurrent_access():
    """Thread safety of binding lookup."""
    any_failed = False
    t3.create_raw_binding()

    def repeatedly_attempt_conversions():
        deadline = time.time() + 1
        while time.time() < deadline:
            try:
                assert t3.check(t3.make(5)) == 5
            except:
                nonlocal any_failed
                any_failed = True
                raise

    threads = [
        threading.Thread(target=repeatedly_attempt_conversions) for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not any_failed


# =====================================================================
# Test 14: Concurrent modification (free-threaded only)
# =====================================================================

@pytest.mark.skipif(not free_threaded, reason="not relevant on non-FT")
def test_concurrent_modification():
    """Thread safety of binding add/remove during lookup."""
    transitions = 0
    limit = 5000

    t1.bind_types()
    t2.bind_types()
    t3.create_raw_binding()

    def repeatedly_remove_and_readd():
        nonlocal transitions
        try:
            while transitions < limit:
                del t3.RawShared.__pymetabind_binding__
                t3.export_raw_binding()
                transitions += 1
        except:
            transitions = limit
            raise

    thread = threading.Thread(target=repeatedly_remove_and_readd)
    thread.start()

    num_failed = 0
    num_successful = 0

    def repeatedly_attempt_conversions():
        nonlocal num_failed
        nonlocal num_successful
        while transitions < limit:
            try:
                assert t3.check(t3.make(42)) == 42
            except TypeError:
                num_failed += 1
            else:
                num_successful += 1

    try:
        threads = [
            threading.Thread(target=repeatedly_attempt_conversions) for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        transitions = limit
        thread.join()

    assert num_successful > 0
    assert num_failed > 0


# =====================================================================
# Test 15: Native enum interop
# =====================================================================

def test_native_enum_interop():
    """SharedEnum can be passed across foreign boundaries."""
    t1.bind_types()
    t2.bind_types()

    # Create enums in each module
    e1 = t1.make_enum(1)
    e2 = t2.make_enum(2)

    # Check locally
    assert t1.check_enum(e1) == 1
    assert t2.check_enum(e2) == 2

    # Check across foreign boundary
    assert t2.check_enum(e1) == 1
    assert t1.check_enum(e2) == 2

    # Verify they are actual enum instances with correct types
    assert type(e1) is t1.SharedEnum
    assert type(e2) is t2.SharedEnum


# =====================================================================
# Test 16: Three-module interop (t1, t2, t3)
# =====================================================================

def test_three_module_interop():
    """Comprehensive test of interop between all three full-mode modules."""
    t1.bind_types()
    t2.bind_types()
    t3.create_raw_binding()

    # t1 <-> t2 works automatically
    expect(t1, t2, "foreign")
    expect(t2, t1, "foreign")

    # t1/t2 -> t3 works automatically for C++ types (including enum)
    expect(t1, t3, "foreign")
    expect(t2, t3, "foreign")

    # t3 -> t1/t2: RawShared is not C++ so value/sp/up don't work,
    # but enum DOES work because t3 auto-imported t1's SharedEnum
    expect(t3, t1, "isolated", enum=True)
    expect(t3, t2, "isolated", enum=True)

    # Import t3.RawShared into both t1 and t2
    t1.import_foreign_explicit(t3.RawShared)
    t2.import_foreign_explicit(t3.RawShared)

    expect(t3, t1, "foreign")
    expect(t3, t2, "foreign")


# =====================================================================
# Test 17: Local type stays preferred over foreign
# =====================================================================

def test_local_preferred_over_foreign():
    """When a module has its own binding for a type, it should always
    prefer the local binding over any foreign ones."""
    t1.bind_types()
    t2.bind_types()

    # t1 creates t1.Shared (its own local type), not t2.Shared
    obj1 = t1.make(10)
    assert type(obj1) is t1.Shared

    # t2 creates t2.Shared (its own local type)
    obj2 = t2.make(20)
    assert type(obj2) is t2.Shared

    # Local check works with local type
    assert t1.check(obj1) == 10
    assert t2.check(obj2) == 20

    # Foreign check also works
    assert t2.check(obj1) == 10
    assert t1.check(obj2) == 20


# =====================================================================
# Test 18: on_request selective import/export
# =====================================================================

def test_on_request_selective():
    """t4 (on_request) can selectively import/export individual types."""
    t1.bind_types()
    t2.bind_types()
    t4.bind_types()

    # Export only t4's Shared, not SharedEnum
    t4.export_to_foreign(t4.Shared)

    # t1 can use t4's Shared (auto-imports) but not enum
    assert t1.check(t4.make(42)) == 42
    with pytest.raises(TypeError):
        t1.check_enum(t4.make_enum(1))

    # Import only t1's Shared into t4, not t2's
    t4.import_foreign(t1.Shared)
    assert t4.check(t1.make(99)) == 99

    # t2's types still can't be consumed by t4
    with pytest.raises(TypeError):
        t4.check(t2.make(50))

    # Import t2's too
    t4.import_foreign(t2.Shared)
    assert t4.check(t2.make(50)) == 50


# =====================================================================
# Test 19: Export only mode (using t1 with manual export check)
# =====================================================================

def test_export_without_import():
    """t1 auto-exports. Verify that auto-export means other modules
    can see t1's types after explicit import."""
    t1.bind_types()
    t4.bind_types()

    # t4 needs explicit import since it's on_request mode
    t4.import_foreign(t1.Shared)
    t4.import_foreign(t1.SharedEnum)

    # t4 can now use t1's types
    expect(t1, t4, "foreign")

    # t1 auto-imports, but t4 hasn't exported
    expect(t4, t1, "isolated")


# =====================================================================
# Test 20: Binding types multiple times is safe
# =====================================================================

def test_bind_types_idempotent():
    """Calling bind_types() multiple times should be safe."""
    t1.bind_types()
    t1.bind_types()  # Should not raise or create duplicates
    t2.bind_types()
    t2.bind_types()

    # Everything still works
    expect(t1, t2, "foreign")
    expect(t2, t1, "foreign")


# =====================================================================
# Test 21: Interop with RawShared (C API framework) value round-trip
# =====================================================================

def test_raw_shared_value_roundtrip():
    """Test that values created by the C API framework (t3.RawShared)
    can be passed through pybind11 functions."""
    t1.bind_types()
    t3.create_raw_binding()
    t1.import_foreign_explicit(t3.RawShared)

    # Create with t3, check with t1
    obj = t3.make(42)
    assert type(obj) is t3.RawShared
    assert t1.check(obj) == 42

    # shared_ptr: foreign creates new control block
    sp = t3.make_sp(99)
    assert t1.check_sp(sp) == 99
    assert t1.uses(sp) == 1  # Foreign shared_ptr -> new control block


# =====================================================================
# Test 22: t3 local binding coexists with foreign
# =====================================================================

def test_local_and_foreign_coexist():
    """t3 can have both its own pybind11 Shared binding AND use foreign
    ones. Local should always be preferred."""
    t1.bind_types()
    t3.create_raw_binding()
    t3.bind_types()  # Now t3 has a pybind11 Shared AND a RawShared

    # t3 prefers its RawShared (registered first with the C API framework)
    # for values created by t3's own make() function.
    # But t3's pybind11 Shared also exists.
    obj = t3.make(10)
    # The local pybind11 binding should be preferred for its own make()
    assert type(obj) is t3.Shared or type(obj) is t3.RawShared
    assert t3.check(obj) == 10

    # t1's types should work in t3 (auto-imported)
    assert t3.check(t1.make(20)) == 20


# =====================================================================
# Test 23: shared_ptr from foreign (share_ownership RVP)
# =====================================================================

def test_shared_ptr_foreign_ownership():
    """When a foreign-bound function returns shared_ptr, pybind11 should
    create a new shared_ptr control block that keeps the Python object alive.
    This tests the foreign_cb_keep_alive share_ownership path."""
    t1.bind_types()
    t2.bind_types()

    # Create a shared_ptr in t1 and pass to t2
    sp = t1.make_sp(42)
    assert type(sp) is t1.Shared
    assert t1.uses(sp) == 2  # local: real shared_ptr

    # Check it via t2 (foreign)
    assert t2.check_sp(sp) == 42
    assert t2.uses(sp) == 1  # foreign: new control block

    # Create shared_ptr in t2 and check via t1
    sp2 = t2.make_sp(99)
    assert t1.check_sp(sp2) == 99
    assert t1.uses(sp2) == 1


# =====================================================================
# Test 24: Export idempotent
# =====================================================================

def test_export_idempotent():
    """Exporting the same type multiple times should be a no-op."""
    t1.bind_types()

    # Export multiple times - should not raise or create duplicates
    t1.export_to_foreign(t1.Shared)
    t1.export_to_foreign(t1.Shared)
    t1.export_to_foreign(t1.SharedEnum)
    t1.export_to_foreign(t1.SharedEnum)

    # t2 should still be able to use t1's types (only one binding)
    t2.bind_types()
    expect(t1, t2, "foreign")


# =====================================================================
# Test 25: Return value policy none (lookup existing instance)
# =====================================================================

def test_rvp_none_foreign():
    """When converting C++ to Python with RVP 'none' via foreign binding,
    pybind11 should return an existing registered instance or None."""
    t1.bind_types()
    t2.bind_types()

    # make() returns by value (creates a new instance each time)
    # make_sp() returns shared_ptr (the underlying C++ object can be found)
    sp = t2.make_sp(77)
    assert t2.check_sp(sp) == 77

    # t1 can also check it
    assert t1.check_sp(sp) == 77


# =====================================================================
# Test 26: Exception translator chain with multiple frameworks
# =====================================================================

@pytest.mark.skipif(
    (env.MACOS and env.PYPY) or env.ANDROID,
    reason="same issue as test_exceptions.py test_cross_module_exception_translator",
)
def test_exception_translator_chain():
    """When multiple frameworks register exception translators, they should
    be tried in order. t2 has a translator for SharedExc; t1 and t3 do not."""
    t1.bind_types()
    t2.bind_types()
    t3.create_raw_binding()

    # t1's throw_shared should be translated by t2's translator
    with pytest.raises(ValueError, match="Shared.42"):
        t1.throw_shared(42)

    # t2's own throw_shared also works
    with pytest.raises(ValueError, match="Shared.99"):
        t2.throw_shared(99)


# =====================================================================
# Test 27: on_request module export makes types visible to full modules
# =====================================================================

def test_on_request_export_visible_to_full():
    """When an on_request module exports a type, full-mode modules
    should automatically see it (they auto-import)."""
    t4.bind_types()
    t4.export_to_foreign(t4.Shared)

    t1.bind_types()  # t1 auto-imports, should pick up t4.Shared

    # t1 should be able to accept t4's Shared
    assert t1.check(t4.make(55)) == 55


# =====================================================================
# Test 28: on_request mode with local types
# =====================================================================

def test_on_request_with_local_binding():
    """t4 (on_request) has its own local binding. It should prefer
    local over foreign, and foreign should work after import."""
    t1.bind_types()
    t4.bind_types()

    # t4 prefers its own local binding
    obj4 = t4.make(10)
    assert type(obj4) is t4.Shared
    assert t4.check(obj4) == 10

    # Import t1's binding into t4
    t4.import_foreign(t1.Shared)

    # t4 still prefers local
    obj4b = t4.make(20)
    assert type(obj4b) is t4.Shared

    # But t4 can now accept t1's types
    obj1 = t1.make(30)
    assert t4.check(obj1) == 30


# =====================================================================
# GraalPy has issues with GC-dependent tests
# =====================================================================

if sys.implementation.name == "graalpy":
    del test_three_module_interop
    del test_implicit_conversion_from_foreign
