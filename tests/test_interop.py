# Copyright (c) 2025 The pybind Community.
from __future__ import annotations

import collections
import gc
import itertools
import sys
import threading
import time
import weakref

import pytest
import test_interop_1 as t1
import test_interop_2 as t2
import test_interop_3 as t3

free_threaded = hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()

# t1, t2, t3 all define bindings for the same C++ type `Shared`, as well as
# several functions to create and inspect instances of that type. Each module
# uses a different PYBIND11_INTERNALS_VERSION, so they won't interoperate
# natively; these tests check the foreign-framework interoperability mechanism.
# The bindings are all different, as follows:
#   t1.Shared uses a std::shared_ptr holder
#   t2.Shared uses a smart_holder
#   t3.RawShared uses the Python C API to mimic a non-pybind11 framework
# Upon import, bindings are defined for the functions, but not for the
# types (`Shared` and `SharedEnum`) until you call bind_types().
#
# NB: there is some potential for different test cases to interfere with
# each other: we can't un-register a framework once it's registered and we
# can't undo automatic import/export all once they're requested. The ordering
# of these tests is therefore important. They work standalone and they work
# when run all in one process, but they might not work in a different order.


def delattr_and_ensure_destroyed(*specs):
    wrs = []
    for mod, name in specs:
        wrs.append(weakref.ref(getattr(mod, name)))
        delattr(mod, name)

    for attempt in range(5):
        gc.collect()
        if all(wr() is None for wr in wrs):
            break
    else:
        pytest.fail(
            f"Could not delete bindings such as {next(wr for wr in wrs if wr() is not None)!r}"
        )


@pytest.fixture(autouse=True)
def clean_after():
    yield
    t3.clear_interop_bindings()

    delattr_and_ensure_destroyed(
        *[
            (mod, name)
            for mod in (t1, t2, t3)
            for name in ("Shared", "SharedEnum", "RawShared")
            if hasattr(mod, name)
        ]
    )

    t1.pull_stats()
    t2.pull_stats()
    t3.pull_stats()


def check_stats(mod, **entries):
    if mod is None:
        return
    if sys.implementation.name == "pypy":
        gc.collect()
    stats = mod.pull_stats()
    for name, value in entries.items():
        assert stats.pop(name) == value
    assert all(val == 0 for val in stats.values())


def test01_interop_exceptions_without_registration():
    # t2 defines the exception translator for Shared. Since it hasn't
    # taken any interop actions yet, it hasn't registered with pymetabind
    # and t1 won't be able to use that translator.
    with pytest.raises(RuntimeError, match="Caught an unknown exception"):
        t1.throw_shared(100)

    with pytest.raises(ValueError, match="Shared.200"):
        t2.throw_shared(200)


global_counter = itertools.count()


def expect(from_mod, to_mod, pattern, **extra):
    outcomes = {}
    extra_info = {}
    owner_mod = None

    for idx, suffix in enumerate(("", "_sp", "_up", "_enum")):
        create = getattr(from_mod, f"make{suffix}")
        check = getattr(to_mod, f"check{suffix}")
        thing = suffix.lstrip("_") or "value"
        print(thing)
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
            # Include shared_ptr use count in the test. Foreign should create
            # a new control block so we see use_count == 1. Local should reuse
            # the same -> use_count == 0.
            outcomes[thing] = to_mod.uses(obj)
        else:
            outcomes[thing] = True

    expected = {}
    if pattern == "local":
        # Accepting a unique_ptr argument only works for non-foreign smart_holder
        expected = {"value": True, "sp": 2, "up": to_mod is t2, "enum": True}
    elif pattern == "foreign":
        expected = {"value": True, "sp": 1, "up": False, "enum": True}
    elif pattern == "isolated":
        expected = {"value": False, "sp": False, "up": False, "enum": False}
    elif pattern == "none":
        expected = {"value": None, "sp": None, "up": None, "enum": None}
    else:
        assert False, "unknown pattern"
    expected.update(extra)
    assert outcomes == expected

    obj = None

    # When returning by value, we have a construction in from_mod,
    # move to owner_mod, destruction in from_mod (after make() returns)
    # and destruction in owner mod (when the pyobject dies).
    #
    # When returning shared_ptr, the construction and destruction both
    # occur in from_mod since shared_ptr's deleter is set at creation time.
    #
    # When returning unique_ptr, the construction occurs in from_mod and
    # destruction (when the pyobject dies) occurs in owner_mod; unless
    # we pass ownership to to_mod, in which case the destruction happens there.
    # But since we can't pass ownership to a foreign framework currently,
    # we'll disregard that possibility and always attribute it to owner_mod.
    expect_stats = {mod: collections.Counter() for mod in (from_mod, to_mod, owner_mod)}
    expect_stats[from_mod].update(
        ["construct", "destroy", "construct", "destroy", "construct"]
    )
    # value move+destroy
    expect_stats[owner_mod].update(["move", "destroy"])
    # unique_ptr destroy; due to an existing pybind11 bug this may be skipped
    # entirely (leaked) if we return a raw pointer with rvp take_ownership and
    # the cast fails
    if owner_mod is None and from_mod is t1:
        pass
    else:
        expect_stats[owner_mod or from_mod].update(["destroy"])
    for mod, stats in expect_stats.items():
        check_stats(mod, **stats)


def test02_interop_unimported():
    # Before any types are bound, no to-Python conversions are possible
    for mod in (t1, t2, t3):
        expect(mod, mod, "none")

    # Bind the types but don't share them yet
    t1.bind_types()
    t2.bind_types()

    for mod in (t1, t2):
        expect(mod, mod, "local")

    # t3 hasn't defined SharedEnum yet. Its version of Shared is not
    # bound using pybind11, so is foreign even to the functions in t3.
    t3.create_raw_binding()
    expect(t3, t3, "foreign", enum=None)

    expect(t1, t2, "isolated")
    expect(t1, t3, "isolated")
    expect(t2, t1, "isolated")
    expect(t2, t3, "isolated")
    expect(t3, t1, "isolated", enum=None)
    expect(t3, t2, "isolated", enum=None)

    # Just an export isn't enough; you need an import too
    t2.export_for_interop(t2.Shared)
    expect(t2, t3, "isolated")


def test03_interop_import_export_errors():
    t1.bind_types()
    t2.bind_types()
    t3.create_raw_binding()

    with pytest.raises(
        RuntimeError, match="type does not define a __pymetabind_binding__"
    ):
        t2.import_for_interop(t1.Convertible)

    with pytest.raises(RuntimeError, match="not a pybind11 class or enum"):
        t3.export_for_interop(t2.Shared)

    with pytest.raises(RuntimeError, match="not a pybind11 class or enum"):
        t3.export_for_interop(t2.SharedEnum)

    t2.export_for_interop(t2.Shared)
    t2.export_for_interop(t2.SharedEnum)
    t2.export_for_interop(t2.Shared)  # should be idempotent
    t2.export_for_interop(t2.SharedEnum)

    with pytest.raises(RuntimeError, match="type is not foreign"):
        t2.import_for_interop(t2.Shared)

    with pytest.raises(RuntimeError, match=r"is not written in C\+\+"):
        t2.import_for_interop(t3.RawShared)

    t2.import_for_interop_explicit(t3.RawShared)
    t2.import_for_interop_explicit(t3.RawShared)  # should be idempotent

    with pytest.raises(
        RuntimeError, match=r"was already imported as a different C\+\+ type"
    ):
        t2.import_for_interop_wrong_type(t3.RawShared)


def test04_interop_exceptions():
    # Once t1 and t2 have registered with pymetabind, which happens as soon as
    # they each import or export anything, t1 can translate t2's exceptions.
    t1.bind_types()
    t2.bind_types()
    t1.export_for_interop(t1.Shared)
    t2.export_for_interop(t2.Shared)
    with pytest.raises(ValueError, match="Shared.123"):
        t1.throw_shared(123)


def test05_interop_with_cpp():
    t1.bind_types()
    t2.bind_types()
    t3.create_raw_binding()

    # Export t1/t2's Shared to t3, but not the enum yet, and not from t3
    t1.export_for_interop(t1.Shared)
    t2.export_for_interop(t2.Shared)
    t3.import_for_interop(t1.Shared)
    t3.import_for_interop(t2.Shared)
    expect(t1, t3, "foreign", enum=False)
    expect(t2, t3, "foreign", enum=False)
    expect(t1, t2, "isolated")
    expect(t3, t1, "isolated", enum=None)
    expect(t3, t2, "isolated", enum=None)

    # Now export t2.SharedEnum too. Note that t3 doesn't have its own
    # definition of SharedEnum yet, so it will use the imported one and create
    # t2.SharedEnums.
    t2.export_for_interop(t2.SharedEnum)
    t3.import_for_interop(t2.SharedEnum)
    expect(t1, t3, "foreign", enum=False)
    expect(t2, t3, "foreign")
    expect(t3, t2, "isolated", enum=True)

    t1.export_for_interop(t1.SharedEnum)
    t3.import_for_interop(t1.SharedEnum)
    expect(t1, t3, "foreign")
    expect(t2, t1, "isolated")  # t1 hasn't imported anything
    expect(t2, t3, "foreign")
    expect(t3, t1, "isolated")  # t3 sends t2.SharedEnums which t1 can't read

    t1.import_for_interop(t2.SharedEnum)
    expect(t2, t1, "isolated", enum=True)
    expect(t3, t1, "isolated", enum=True)

    t1.import_for_interop(t2.Shared)
    expect(t2, t1, "foreign")

    # No one has imported t3.RawShared, so t3->X doesn't work yet

    # t1/t2 each return their local type since it exists (local is always
    # preferred). t3 returns its non-pybind extension type because it has
    # an import for that one before any of the imports we wrote.
    assert type(t1.make(1)) is t1.Shared
    assert type(t2.make(2)) is t2.Shared
    assert type(t3.make(3)) is t3.RawShared

    # If we create a pybind11 Shared in t3, that takes priority over the raw one
    t3.bind_types()
    assert type(t3.make(4)) is t3.Shared


def test06_interop_return_foreign_smart_holder():
    # Test a pybind11 domain returning a different pybind11 domain's type
    # because it didn't have its own.
    t2.bind_types()
    t2.export_for_interop(t2.Shared)
    t2.export_for_interop(t2.SharedEnum)
    t1.import_for_interop(t2.Shared)
    t1.import_for_interop(t2.SharedEnum)
    expect(t2, t1, "foreign")
    expect(t1, t2, "local")  # t2 is both source and dest
    assert type(t1.make(1)) is t2.Shared
    assert type(t2.make(2)) is t2.Shared


def test06_interop_return_foreign_shared_ptr():
    t1.bind_types()
    t1.export_for_interop(t1.Shared)
    t1.export_for_interop(t1.SharedEnum)
    t2.import_for_interop(t1.Shared)
    t2.import_for_interop(t1.SharedEnum)
    expect(t1, t2, "foreign")
    expect(t2, t1, "local")  # t1 is both source and dest
    assert type(t1.make(1)) is t1.Shared
    assert type(t2.make(2)) is t1.Shared


def test07_interop_with_c():
    t1.bind_types()
    t3.create_raw_binding()
    t1.export_for_interop(t1.SharedEnum)
    t3.import_for_interop(t1.SharedEnum)
    t1.import_for_interop_explicit(t3.RawShared)

    # Now that t3.RawShared is imported to t1, we can send in the t3->t1 direction.
    expect(t3, t1, "foreign")


def test08_remove_binding():
    t3.create_raw_binding()
    t2.import_for_interop_explicit(t3.RawShared)

    # Remove the binding for t3.RawShared. We expect the t2 domain will
    # notice the removal and automatically forget about the defunct binding.
    delattr_and_ensure_destroyed((t3, "RawShared"))
    t3.create_raw_binding()

    t2.bind_types()
    t2.export_for_interop(t2.Shared)
    t3.import_for_interop(t2.Shared)
    t2.export_for_interop(t2.SharedEnum)
    t3.import_for_interop(t2.SharedEnum)

    expect(t2, t3, "foreign")
    expect(t3, t2, "isolated", enum=True)

    # Similarly test removal of t2.Shared / t2.SharedEnum.
    delattr_and_ensure_destroyed((t2, "Shared"), (t2, "SharedEnum"))
    t2.bind_types()

    expect(t2, t3, "isolated")
    expect(t3, t2, "isolated", enum=None)

    t2.export_for_interop(t2.Shared)
    t3.import_for_interop(t2.Shared)
    t2.export_for_interop(t2.SharedEnum)
    t3.import_for_interop(t2.SharedEnum)

    expect(t2, t3, "foreign")
    expect(t3, t2, "isolated", enum=True)

    # Removing the binding capsule should work just as well as removing
    # the type object.
    del t2.Shared.__pymetabind_binding__
    del t2.SharedEnum.__pymetabind_binding__
    gc.collect()

    expect(t2, t3, "isolated")
    expect(t3, t2, "isolated", enum=None)

    t2.export_for_interop(t2.Shared)
    t3.import_for_interop(t2.Shared)
    t2.export_for_interop(t2.SharedEnum)
    t3.import_for_interop(t2.SharedEnum)

    expect(t2, t3, "foreign")
    expect(t3, t2, "isolated", enum=True)

    # Re-import RawShared and now everything works again.
    t2.import_for_interop_explicit(t3.RawShared)
    expect(t2, t3, "foreign")
    expect(t3, t2, "foreign")

    # Removing the binding capsule should work just as well as removing
    # the type object.
    del t3.RawShared.__pymetabind_binding__
    gc.collect()
    t3.export_raw_binding()

    # t3.RawShared was removed from the beginning of t3's list for Shared
    # and re-added on the end; also remove and re-add t2.Shared so that
    # t3.make() continues to return a t3.RawShared
    del t2.Shared.__pymetabind_binding__
    t2.export_for_interop(t2.Shared)
    t3.import_for_interop(t2.Shared)

    expect(t2, t3, "foreign")
    expect(t3, t2, "isolated", enum=True)

    # Re-import RawShared and now everything works again.
    t2.import_for_interop_explicit(t3.RawShared)
    expect(t2, t3, "foreign")
    expect(t3, t2, "foreign")


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test09_access_binding_concurrently():
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


@pytest.mark.skipif(not free_threaded, reason="not relevant on non-FT")
def test10_remove_binding_concurrently():
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

    # typical numbers from my machine: with limit=5000,
    # num_failed and num_successful are each several 10k's
    print(num_failed, num_successful)
    assert num_successful > 0
    assert num_failed > 0


def test11_implicit():
    # Create four different types of pyobject, all of which have C++ type Shared
    t1.bind_types()
    t2.bind_types()
    t3.create_raw_binding()
    s1 = t1.make(10)
    s2 = t2.make(11)
    s3r = t3.make(12)
    t3.bind_types()
    s3p = t3.make(13)

    t2.export_all()
    t3.export_all()
    t1.import_all()
    t1.import_for_interop_explicit(t3.RawShared)

    assert type(s1) is t1.Shared
    assert type(s2) is t2.Shared
    assert type(s3r) is t3.RawShared
    assert type(s3p) is t3.Shared

    # Test implicit conversions from foreign types
    for idx, obj in enumerate((s1, s2, s3r, s3p)):
        val = t1.test_implicit(obj)
        assert val == 10 + idx

    # We should only be sharing in the tX->t1 direction, not vice versa
    assert t1.check(s2) == 11
    with pytest.raises(TypeError):
        t2.check(s1)

    # Now add the other direction
    t1.export_all()
    t2.import_all()
    assert t2.check(s1) == 10


def test12_import_export_all():
    # Enable automatic import and export in the t1/t2 domains.
    # Still doesn't help with t3->t1/t2 since t3.RawShared is not a C++ type.
    t1.import_all()
    t1.export_all()
    t1.bind_types()

    t2.import_all()
    t2.bind_types()
    t2.export_all()

    t3.create_raw_binding()
    t3.import_for_interop(t1.Shared)
    t3.import_for_interop(t2.SharedEnum)

    expect(t1, t2, "foreign")
    expect(t1, t3, "foreign", enum=False)  # t3 didn't import all or import t1's enum
    expect(t2, t1, "foreign")
    expect(t2, t3, "isolated", enum=True)  # t3 didn't import t2.Shared
    t3.import_all()
    expect(t2, t3, "foreign")  # t3 didn't import t2.Shared
    expect(t3, t1, "isolated", enum=True)
